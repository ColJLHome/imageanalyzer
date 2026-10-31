import base64
from contextlib import contextmanager
import io
import json
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
import os
from pathlib import Path
import sqlite3
import threading
import time

import httpx
from PIL import Image, ImageOps

from .settings import RuntimeSettings

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
PROMPT = '''Describe only what is visibly present. Return one JSON object with:
"description": a short factual description,
"scenes": an array of scene names,
"objects": an array of objects including their visible colours (e.g. "red car"),
"colours": an array of ordinary colour names.
Do not infer identities, locations, or invisible facts. No markdown.'''


ANALYSIS_SCHEMA = {
    'type': 'object',
    'properties': {
        'description': {'type': 'string'},
        **{field: {'type': 'array', 'items': {'type': 'string'}}
           for field in ('scenes', 'objects', 'colours')},
    },
    'required': ['description', 'scenes', 'objects', 'colours'],
    'additionalProperties': False,
}


def parse_analysis(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Model returned no final answer content')
    content = content.strip()
    if content.startswith('<think>'):
        _, end, content = content.partition('</think>')
        if not end:
            raise ValueError('Model returned unfinished thinking without a final answer')
        content = content.strip()
    if content.startswith('```'):
        _, newline, content = content.partition('\n')
        if not newline:
            raise ValueError('Model returned an incomplete markdown block')
        content = content.rsplit('```', 1)[0].strip()
    data = json.loads(content)
    if not isinstance(data, dict) or not isinstance(data.get('description'), str):
        raise ValueError('Model response must contain a description string')
    result = {'description': data['description'][:8000]}
    for field in ('scenes', 'objects', 'colours'):
        values = data.get(field)
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f'Model response must contain a {field} string array')
        result[field] = [v[:200].strip().lower() for v in values[:100] if v.strip()]
    return result


class Index:
    def __init__(self, config):
        self.config = dict(config)
        self.roots = [Path(p).expanduser().resolve() for p in config['roots']]
        if not self.roots:
            raise ValueError('Configure at least one image root')
        if config.get('scan_interval_seconds', 3600) < 1:
            raise ValueError('scan_interval_seconds must be positive')
        self.data = Path(config.get('data_dir', 'data')).resolve()
        self.data.mkdir(parents=True, exist_ok=True)
        self.database = self.data / 'index.sqlite3'
        self.log_path = self.data / 'imageanalyser.log'
        self.log = logging.Logger('imageanalyser.index')
        self.log.parent = logging.getLogger('imageanalyser')
        self.log.setLevel(logging.INFO)
        self.log_handler = RotatingFileHandler(self.log_path, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.log.addHandler(self.log_handler)
        self.control_lock = threading.RLock()
        self.generation = 0
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.scan_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {'running': False, 'current': None, 'roots': {}, 'last_scan': None}
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)')
            saved = db.execute("SELECT value FROM settings WHERE key='runtime_config'").fetchone()
            if saved:
                self.config.update(RuntimeSettings.model_validate_json(saved['value']).model_dump())
                self.roots = [Path(p) for p in self.config['roots']]
            paused = db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()
            self.paused = paused is not None and paused['value'] == 'true'
            db.execute('''CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                size INTEGER, mtime_ns INTEGER, status TEXT NOT NULL DEFAULT 'pending',
                analysis TEXT, search_text TEXT, palette TEXT, width INTEGER, height INTEGER,
                model TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                retry_at REAL NOT NULL DEFAULT 0, updated REAL)''')

    def get_settings(self):
        with self.control_lock:
            return RuntimeSettings(**{key: self.config[key] for key in RuntimeSettings.model_fields
                                      if key in self.config}).model_dump()

    def save_settings(self, settings):
        values = settings.model_dump()
        with self.control_lock:
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO settings VALUES ('runtime_config', ?)",
                           (json.dumps(values),))
                db.execute("UPDATE images SET retry_at=0,attempts=0 WHERE status='error'")
            self.config = {**self.config, **values}
            self.roots = [Path(p) for p in values['roots']]
            self.generation += 1
        self.log.info('Settings saved; %s scan folders configured', len(values['roots']))
        self.wake.set()
        return values

    def clear(self):
        with self.control_lock:
            with self.connect() as db:
                count = db.execute('SELECT count(*) FROM images').fetchone()[0]
                db.execute('DELETE FROM images')
                db.execute("INSERT OR REPLACE INTO settings VALUES ('paused', 'true')")
            self.paused = True
            self.generation += 1
        self.log.info('Cleared %s image records; scanning paused until Resume scanning', count)
        return count

    def remove_image(self, image_id):
        with self.control_lock:
            with self.connect() as db:
                row = db.execute("SELECT path FROM images WHERE id=? AND status!='removed'",
                                 (image_id,)).fetchone()
                if not row:
                    return False
                # Retain only file identity and an exclusion marker so scans skip it.
                db.execute("""UPDATE images SET status='removed',analysis=NULL,search_text=NULL,
                    palette=NULL,width=NULL,height=NULL,model=NULL,error=NULL,attempts=0,
                    retry_at=0,updated=? WHERE id=?""", (time.time(), image_id))
        self.log.info('Removed image from index; excluded from future scans: %s', row['path'])
        return True

    def clear_scanner_details(self):
        with self.state_lock:
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO settings VALUES ('diagnostics_cleared_at', ?)",
                           (str(time.time()),))
            self.state.update(current=None, roots={}, last_scan=None)
        self.log.info('Scanner details cleared by user')

    def resume(self):
        with self.control_lock:
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO settings VALUES ('paused', 'false')")
            self.paused = False
        self.log.info('Scanning resumed by user')
        self.wake.set()

    def request_scan(self):
        with self.control_lock:
            if not self.paused and self.scan_lock.locked():
                return 'running'
            self.resume()
            self.log.info('Scan for new images requested by user')
            return 'queued'

    def logs(self, limit=300):
        # Bound disk reads and responses even when the log file is large.
        with self.log_path.open('rb') as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 256_000)
            stream.seek(start)
            if start:
                stream.readline()
            return list(deque(stream.read().decode('utf-8', errors='replace').splitlines(), maxlen=limit))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def permitted(self, path):
        path = Path(path).resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError('Image is outside configured roots')
        return path

    def prepare(self, path):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            width, height = image.size
            image.thumbnail((self.config.get('max_image_edge', 1280),) * 2)
            small = image.copy()
            small.thumbnail((128, 128))
            quantized = small.quantize(colors=5)
            palette = quantized.getpalette()
            total = small.width * small.height
            colours = [{'hex': '#%02x%02x%02x' % tuple(palette[i*3:i*3+3]),
                        'fraction': round(count / total, 3)}
                       for count, i in sorted(quantized.getcolors(), reverse=True)]
            out = io.BytesIO()
            image.save(out, format='JPEG', quality=85)
            return out.getvalue(), colours, width, height

    def analyse(self, path):
        with self.control_lock:
            config = dict(self.config)
        payload, palette, width, height = self.prepare(path)
        headers = {}
        if os.environ.get('LLAMA_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['LLAMA_API_KEY']
        response = httpx.post(config['llama_url'].rstrip('/') + '/chat/completions',
            headers=headers, timeout=config.get('request_timeout_seconds', 180),
            json={'model': config.get('model', 'local-vision'), 'temperature': 0.1,
                  'max_tokens': config.get('max_output_tokens', 2048), 'stream': False,
                  'response_format': {'type': 'json_object', 'schema': ANALYSIS_SCHEMA},
                  'chat_template_kwargs': {'enable_thinking': config.get('enable_thinking', False)},
                  'messages': [{'role': 'user', 'content': [
                      {'type': 'text', 'text': PROMPT},
                      {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' +
                          base64.b64encode(payload).decode()}}]}]})
        response.raise_for_status()
        data = response.json()
        try:
            choice = data['choices'][0]
            message = choice['message']
            content = message.get('content')
            if choice.get('finish_reason') == 'length':
                raise ValueError('Model hit its output token limit before completing the answer; '
                                 'disable thinking or increase max_output_tokens')
            result = parse_analysis(content)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            # Log bounded final text and metadata, never image data or request headers.
            choice = data.get('choices') if isinstance(data, dict) else None
            choice = choice[0] if isinstance(choice, list) and choice and isinstance(choice[0], dict) else {}
            message = choice.get('message')
            message = message if isinstance(message, dict) else {}
            content = message.get('content')
            reasoning = message.get('reasoning_content') or message.get('reasoning')
            finish = choice.get('finish_reason', 'unknown')
            self.log.error('Invalid model response: finish_reason=%s; content_preview=%r; '
                           'reasoning_chars=%s; usage=%r', str(finish)[:80],
                           content[:500] if isinstance(content, str) else type(content).__name__,
                           len(reasoning) if isinstance(reasoning, str) else 0,
                           str(data.get('usage', {}))[:500] if isinstance(data, dict) else 'unknown')
            raise ValueError(f'Invalid model answer (finish_reason={str(finish)[:80]}): {exc}') from exc
        return result, palette, width, height

    def scan(self):
        if self.paused or not self.scan_lock.acquire(blocking=False):
            return
        with self.control_lock:
            scan_generation = self.generation
            roots = list(self.roots)
        with self.state_lock:
            self.state.update(running=True, roots={})
        self.log.info('Scan started')
        try:
            for root in roots:
                if self.stop.is_set() or self.paused or scan_generation != self.generation:
                    break
                if not root.is_dir():
                    self.log.warning('Root unavailable: %s', root)
                    with self.state_lock:
                        self.state['roots'][str(root)] = 'Unavailable; index retained'
                    continue
                self.log.info('Scanning root: %s', root)
                errors = []
                with self.state_lock:
                    self.state['roots'][str(root)] = 'Scanning'
                for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
                    dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
                    for name in files:
                        if self.stop.is_set() or self.paused or scan_generation != self.generation:
                            return
                        candidate = Path(directory) / name
                        if candidate.suffix.lower() not in EXTENSIONS or candidate.is_symlink():
                            continue
                        try:
                            self.process(self.permitted(candidate), scan_generation)
                        except Exception:
                            self.log.exception('Cannot index %s', candidate)
                for error in errors:
                    self.log.warning('Folder traversal failed: %s', error)
                with self.state_lock:
                    self.state['roots'][str(root)] = '; '.join(errors) if errors else 'Scanned'
        finally:
            with self.state_lock:
                self.state.update(running=False, current=None, last_scan=time.time())
            self.scan_lock.release()
            self.log.info('Scan finished%s', ' (paused)' if self.paused else '')

    def process(self, path, expected_generation=None):
        with self.control_lock:
            if self.paused or (expected_generation is not None and expected_generation != self.generation):
                return
            generation = self.generation
            model = self.config.get('model', 'local-vision')
            stat = path.stat()
            with self.connect() as db:
                row = db.execute('SELECT * FROM images WHERE path=?', (str(path),)).fetchone()
                if row and row['status'] == 'removed':
                    return
                unchanged = row and (row['size'], row['mtime_ns']) == (stat.st_size, stat.st_mtime_ns)
                if unchanged and (row['status'] == 'done' or row['retry_at'] > time.time()):
                    return
                attempts = row['attempts'] if unchanged else 0
                db.execute('''INSERT INTO images(path,size,mtime_ns) VALUES(?,?,?)
                    ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,
                    status='pending',analysis=NULL,search_text=NULL,palette=NULL,error=NULL''',
                    (str(path), stat.st_size, stat.st_mtime_ns))
        with self.state_lock:
            self.state['current'] = str(path)
        self.log.info('Analysing image: %s', path)
        started = time.monotonic()
        try:
            result, palette, width, height = self.analyse(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError('File changed during analysis; will retry')
            search = ' '.join([path.name, result['description'], *result['scenes'],
                               *result['objects'], *result['colours']]).lower()
            with self.control_lock:
                if generation != self.generation:
                    self.log.info('Discarded analysis after index or settings changed: %s', path)
                    return
                with self.connect() as db:
                    updated = db.execute('''UPDATE images SET status='done',analysis=?,search_text=?,palette=?,
                        width=?,height=?,model=?,error=NULL,attempts=0,retry_at=0,updated=? WHERE path=? AND status!='removed' ''',
                        (json.dumps(result), search, json.dumps(palette), width, height,
                         model, time.time(), str(path)))
                    if not updated.rowcount:
                        self.log.info('Discarded analysis for removed image: %s', path)
                        return
            self.log.info('Indexed image in %.1fs: %s', time.monotonic() - started, path)
        except Exception as exc:
            self.log.exception('Analysis failed for %s', path)
            with self.control_lock:
                if generation != self.generation:
                    return
                with self.connect() as db:
                    db.execute("UPDATE images SET status='error',error=?,attempts=?,retry_at=?,updated=? WHERE path=? AND status!='removed'",
                        (str(exc)[:1000], attempts + 1, time.time() + min(3600, 60 * 2 ** min(attempts, 6)), time.time(), str(path)))

    def run(self):
        while not self.stop.is_set():
            self.wake.clear()
            try:
                self.scan()
            except Exception:
                self.log.exception('Scanner failed; will retry on next scan')
            self.wake.wait(self.config.get('scan_interval_seconds', 3600))

    def search(self, q='', scene='', object_name='', colour='', offset=0, limit=60):
        clauses, params = ["status='done'"], []
        for term in q.lower().split():
            clauses.append("instr(search_text, ?) > 0")
            params.append(term)
        for field, value in [('scenes', scene), ('objects', object_name), ('colours', colour)]:
            if value.strip():
                clauses.append(f"EXISTS (SELECT 1 FROM json_each(images.analysis, '$.{field}') WHERE instr(lower(value), ?) > 0)")
                params.append(value.strip().lower())
        where = ' AND '.join(clauses)
        with self.connect() as db:
            total = db.execute('SELECT count(*) FROM images WHERE ' + where, params).fetchone()[0]
            rows = db.execute('SELECT * FROM images WHERE ' + where + ' ORDER BY id DESC LIMIT ? OFFSET ?',
                              [*params, limit, offset]).fetchall()
        return {'total': total, 'items': [dict(id=r['id'], path=r['path'], width=r['width'], height=r['height'],
                   analysis=json.loads(r['analysis']), palette=json.loads(r['palette'])) for r in rows]}
