import base64
import hashlib
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
import tempfile
import time

import httpx
from PIL import Image, ImageOps

from .settings import RuntimeSettings
from .file_order import creation_time_ns
from .metadata import read_a1111_metadata, parse_a1111_text
from .embeddings import OpenClipEncoder, OpenClipError, model_key, rank_vectors
from .diagnostics import CategoryFilter, read_events
from .recognition import RecognizeAnything, RecognitionError, recognition_key, validate_recognition_data

ANALYSIS_VERSION = 1

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
PROMPT = '''Describe only what is visibly present. Return one JSON object with:
"description": a short factual description,
"scenes": an array of scene names,
"objects": an array of objects including their visible colours (e.g. "red car"),
"colours": an array of ordinary colour names,
"people": an array describing up to 10 prominent visible people, or [] if none.
For each person include "position" (e.g. left, centre, foreground), "clothing" (array of
visible garment types, colours, patterns, layers, fit and footwear), "accessories" (array),
"age_group" (child, teenager, young adult, adult, older adult, or unclear), and
"gender_presentation" (masculine, feminine, androgynous, or unclear).
Use approximate apparent age groups only. Describe visible gender presentation, not gender
identity or sex. Use unclear when visibility or appearance does not support an estimate.
Do not guess hidden clothing or specific brands. Keep each person's details concise.
Do not infer identities, locations, or invisible facts. No markdown.'''


AGE_GROUPS = ('child', 'teenager', 'young adult', 'adult', 'older adult', 'unclear')
PRESENTATIONS = ('masculine', 'feminine', 'androgynous', 'unclear')
PERSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'position': {'type': 'string'},
        'clothing': {'type': 'array', 'items': {'type': 'string'}},
        'accessories': {'type': 'array', 'items': {'type': 'string'}},
        'age_group': {'type': 'string', 'enum': list(AGE_GROUPS)},
        'gender_presentation': {'type': 'string', 'enum': list(PRESENTATIONS)},
    },
    'required': ['position', 'clothing', 'accessories', 'age_group', 'gender_presentation'],
    'additionalProperties': False,
}

ANALYSIS_SCHEMA = {
    'type': 'object',
    'properties': {
        'description': {'type': 'string'},
        **{field: {'type': 'array', 'items': {'type': 'string'}}
           for field in ('scenes', 'objects', 'colours')},
        'people': {'type': 'array', 'items': PERSON_SCHEMA, 'maxItems': 10},
    },
    'required': ['description', 'scenes', 'objects', 'colours', 'people'],
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
    # Older stored analyses remain valid; new requests require the people field.
    if 'people' in data:
        if not isinstance(data['people'], list):
            raise ValueError('Model response people must be an array')
        people = []
        for person in data['people'][:10]:
            if not isinstance(person, dict) or not isinstance(person.get('position'), str):
                raise ValueError('Each person must include a position string')
            entry = {'position': person['position'][:200].strip().lower()}
            for field, allowed in (('age_group', AGE_GROUPS), ('gender_presentation', PRESENTATIONS)):
                value = person.get(field)
                if not isinstance(value, str) or value.strip().lower() not in allowed:
                    raise ValueError(f'Each person must include a supported {field} estimate')
                entry[field] = value.strip().lower()
            for field in ('clothing', 'accessories'):
                values = person.get(field)
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise ValueError(f'Each person must include a {field} string array')
                entry[field] = [v[:300].strip().lower() for v in values[:30] if v.strip()]
            people.append(entry)
        result['people'] = people
    return result


class Index:
    def __init__(self, config):
        self.base_config = dict(config)
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
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(category)s] %(message)s'))
        self.log_handler.addFilter(CategoryFilter())
        self.log.addHandler(self.log_handler)
        self.control_lock = threading.RLock()
        self.generation = 0
        self.encoder = OpenClipEncoder()
        self.recognizer = RecognizeAnything()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.scan_lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.state = {'running': False, 'current': None, 'roots': {}, 'last_scan': None,
                      'progress': {'phase': 'idle', 'scanned': 0, 'total': None, 'discovered': 0}}
        self.reset_analysis_timing()
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
            self.migrate_images(db)
            removed = self.purge_excluded_images(db)
        if removed:
            self.log.info('Removed %s records from hidden folders from index', removed)

    @staticmethod
    def migrate_images(db):
        columns = {row['name'] for row in db.execute('PRAGMA table_info(images)')}
        for column, definition in [('sha256', 'TEXT'), ('a1111_metadata', 'TEXT'),
                                   ('analysis_version', 'INTEGER NOT NULL DEFAULT 0'),
                                   ('image_embedding', 'BLOB'), ('embedding_model', 'TEXT'),
                                   ('embedding_sha256', 'TEXT'), ('embedding_error', 'TEXT'),
                                   ('embedding_retry_at', 'REAL NOT NULL DEFAULT 0'),
                                   ('manually_edited', 'INTEGER NOT NULL DEFAULT 0'),
                                   ('ram_data', 'TEXT'), ('ram_model', 'TEXT'), ('ram_sha256', 'TEXT'),
                                   ('ram_error', 'TEXT'), ('ram_retry_at', 'REAL NOT NULL DEFAULT 0'),
                                   ('ram_manually_edited', 'INTEGER NOT NULL DEFAULT 0'),
                                   ('last_indexed_at', 'REAL'), ('created_at_ns', 'INTEGER'),
                                   ('creation_date_fallback', 'INTEGER')]:
            if column not in columns:
                db.execute(f'ALTER TABLE images ADD COLUMN {column} {definition}')
        if 'last_indexed_at' not in columns:
            db.execute("UPDATE images SET last_indexed_at=updated WHERE status='done'")
        db.execute('CREATE INDEX IF NOT EXISTS images_last_indexed ON images(last_indexed_at)')
        db.execute('CREATE INDEX IF NOT EXISTS images_created_at ON images(created_at_ns)')
        db.execute('CREATE INDEX IF NOT EXISTS images_sha256 ON images(sha256)')
        db.execute('CREATE INDEX IF NOT EXISTS images_embedding_hash ON images(embedding_model,embedding_sha256)')
        db.execute('CREATE INDEX IF NOT EXISTS images_ram_hash ON images(ram_sha256,ram_model)')

    @staticmethod
    def excluded_directory(path):
        return any(part.startswith('.') and part not in ('.', '..') for part in Path(path).parts)

    @staticmethod
    def excluded_path(path):
        return Index.excluded_directory(Path(path).parent)

    @staticmethod
    def purge_excluded_images(db):
        # The final slash ensures a dot-prefixed filename alone is not a folder exclusion.
        return db.execute("DELETE FROM images WHERE path GLOB '*/.*/*'").rowcount

    @staticmethod
    def mark_indexed(db, path, metadata=None):
        created = fallback = None
        try:
            created, fallback = creation_time_ns(path, metadata or path.stat())
        except (OSError, ValueError):
            pass
        db.execute("""UPDATE images SET last_indexed_at=updated,
            created_at_ns=coalesce(?,created_at_ns),
            creation_date_fallback=CASE WHEN ? IS NULL THEN creation_date_fallback ELSE ? END
            WHERE path=? AND status='done'""", (created, created, fallback, str(path)))

    def ensure_creation_dates(self):
        with self.control_lock:
            generation = self.generation
            with self.connect() as db:
                paths = [r['path'] for r in db.execute(
                    "SELECT path FROM images WHERE status='done' AND created_at_ns IS NULL")]
        updates = []
        for value in paths:
            try:
                path = self.permitted(value)
                created, fallback = creation_time_ns(path, path.stat())
                updates.append((created, fallback, value))
            except (OSError, ValueError):
                continue
        with self.control_lock:
            if generation == self.generation and updates:
                with self.connect() as db:
                    db.executemany("""UPDATE images SET created_at_ns=?,creation_date_fallback=?
                        WHERE path=? AND status='done' AND created_at_ns IS NULL""", updates)

    def reset_analysis_timing(self):
        self.analysis_count = 0
        self.analysis_seconds = 0.0
        self.active_analysis_started = None

    def progress_snapshot(self):
        with self.state_lock:
            progress = dict(self.state['progress'])
            average = self.analysis_seconds / self.analysis_count if self.analysis_count else None
            remaining = max(0, progress['total'] - progress['scanned']) if progress['total'] is not None else None
            estimate = None
            if progress['phase'] in ('scanning', 'embedding', 'recognizing', 'complete') and remaining is not None:
                if remaining == 0:
                    estimate = 0.0
                elif average is not None:
                    elapsed = (max(0, time.monotonic() - self.active_analysis_started)
                               if self.active_analysis_started is not None else 0)
                    estimate = max(0, remaining * average - min(elapsed, average))
            return {**progress, 'analysis_count': self.analysis_count,
                    'average_analysis_seconds': average, 'estimated_remaining_seconds': estimate}

    def backup_database(self, directory=None, prefix='imageanalyser-backup-'):
        fd, filename = tempfile.mkstemp(prefix=prefix, suffix='.sqlite3', dir=directory)
        os.close(fd)
        destination = Path(filename)
        try:
            with self.control_lock, self.connect() as source:
                target = sqlite3.connect(destination)
                try:
                    source.backup(target)
                    target.execute('PRAGMA journal_mode=DELETE')
                finally:
                    target.close()
            self.log.info('Database backup created')
            return destination
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def restore_database(self, path):
        source = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        source.row_factory = sqlite3.Row
        try:
            source.execute('PRAGMA trusted_schema=OFF')
            if source.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Database failed SQLite integrity check')
            objects = source.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
            if {r['name'] for r in objects if r['type'] == 'table'} != {'images', 'settings'} or any(
                    r['type'] not in ('table', 'index') for r in objects):
                raise ValueError('Upload an Image Analyser SQLite backup')
            columns = {r['name']: r for r in source.execute('PRAGMA table_info(images)')}
            required = {'id', 'path', 'size', 'mtime_ns', 'status', 'analysis', 'search_text', 'palette',
                        'width', 'height', 'model', 'error', 'attempts', 'retry_at', 'updated', 'sha256'}
            if not required <= columns.keys() or columns['id']['pk'] != 1:
                raise ValueError('Backup has an unsupported image schema; use a backup from this version')
            unique_path = any(
                [r['name'] for r in source.execute('SELECT name FROM pragma_index_info(?)', (idx['name'],))] == ['path']
                for idx in source.execute("SELECT name FROM pragma_index_list('images') WHERE [unique]=1"))
            if not unique_path:
                raise ValueError('Backup is missing the unique image path index')
            setting_columns = {r['name']: r for r in source.execute('PRAGMA table_info(settings)')}
            if not {'key', 'value'} <= setting_columns.keys() or setting_columns['key']['pk'] != 1:
                raise ValueError('Backup has an unsupported settings schema')
            saved = source.execute("SELECT value FROM settings WHERE key='runtime_config'").fetchone()
            restored_config = {**self.base_config, **(RuntimeSettings.model_validate_json(saved['value']).model_dump() if saved else {})}
            restored_settings = RuntimeSettings(**{key: restored_config[key] for key in RuntimeSettings.model_fields
                                                  if key in restored_config})
            for row in source.execute('SELECT * FROM images'):
                if 'ram_data' in columns and row['ram_data']:
                    validate_recognition_data(json.loads(row['ram_data']))
                if row['status'] not in ('pending', 'done', 'error', 'removed'):
                    raise ValueError('Backup contains an invalid image status')
                if row['status'] == 'done':
                    parse_analysis(row['analysis'])
                    if not isinstance(json.loads(row['palette']), list):
                        raise ValueError('Backup contains an invalid image palette')
            with self.control_lock:
                backups = self.data / 'backups'
                backups.mkdir(exist_ok=True)
                safety = self.backup_database(backups, 'before-restore-')
                self.paused = True
                self.generation += 1
                with self.connect() as target:
                    source.backup(target)
                    target.execute("INSERT OR REPLACE INTO settings VALUES ('paused', 'true')")
                    self.migrate_images(target)
                    self.purge_excluded_images(target)
                self.config = {**restored_config, **restored_settings.model_dump()}
                self.roots = [Path(p) for p in restored_settings.roots]
            with self.state_lock:
                self.state.update(current=None, roots={}, last_scan=None)
                self.state['progress'] = {'phase': 'idle', 'scanned': 0, 'total': None, 'discovered': 0}
                self.reset_analysis_timing()
            self.log.info('Database restored; scanning paused. Safety backup: %s', safety)
            return safety.name
        except (sqlite3.DatabaseError, TypeError, KeyError) as exc:
            raise ValueError(f'Invalid Image Analyser database: {exc}') from exc
        finally:
            source.close()

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
                db.execute('UPDATE images SET embedding_retry_at=0,embedding_error=NULL,ram_retry_at=0,ram_error=NULL')
            self.config = {**self.config, **values}
            self.roots = [Path(p) for p in values['roots']]
            self.generation += 1
        with self.state_lock:
            self.state['progress'] = {'phase': 'idle', 'scanned': 0, 'total': None, 'discovered': 0}
            self.reset_analysis_timing()
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
        with self.state_lock:
            self.state['progress']['phase'] = 'paused'
        self.log.info('Cleared %s image records; scanning paused until Resume scanning', count)
        return count

    def edit_image(self, image_id, analysis, metadata_raw, ram_tags=None):
        result = parse_analysis(json.dumps(analysis))
        metadata = parse_a1111_text(metadata_raw)
        with self.control_lock:
            with self.connect() as db:
                row = db.execute("SELECT * FROM images WHERE id=? AND status!='removed'", (image_id,)).fetchone()
                if row is None:
                    return False
                ram_data = self.ram_for_hash(row, row['sha256'])
                if ram_tags is not None:
                    if not isinstance(ram_tags, list) or any(not isinstance(tag, str) for tag in ram_tags):
                        raise ValueError('RAM tags must be an array of strings')
                    ram_data = {'tags': list(dict.fromkeys(tag.strip().lower()[:300] for tag in ram_tags[:500] if tag.strip())),
                                'tags_chinese': []}
                    db.execute("""UPDATE images SET ram_data=?,ram_model=coalesce(ram_model,'manual'),
                        ram_sha256=sha256,ram_manually_edited=1,ram_error=NULL,ram_retry_at=0 WHERE id=?""",
                        (json.dumps(ram_data), image_id))
                db.execute("""UPDATE images SET status='done',analysis=?,search_text=?,a1111_metadata=?,
                    manually_edited=1,palette=coalesce(palette,'[]'),analysis_version=?,error=NULL,retry_at=0,attempts=0,updated=? WHERE id=?""",
                    (json.dumps(result), self.search_text(Path(row['path']), result, metadata, ram_data),
                     json.dumps(metadata) if metadata else None, ANALYSIS_VERSION, time.time(), image_id))
                self.mark_indexed(db, Path(row['path']))
        self.log.info('Image data manually corrected: %s', row['path'])
        return True

    def remove_image(self, image_id):
        with self.control_lock:
            with self.connect() as db:
                row = db.execute("SELECT path FROM images WHERE id=? AND status!='removed'",
                                 (image_id,)).fetchone()
                if not row:
                    return False
                # Retain only file identity and an exclusion marker so scans skip it.
                db.execute("""UPDATE images SET status='removed',analysis=NULL,search_text=NULL,
                    palette=NULL,width=NULL,height=NULL,model=NULL,a1111_metadata=NULL,analysis_version=0,error=NULL,attempts=0,
                    retry_at=0,image_embedding=NULL,embedding_model=NULL,embedding_sha256=NULL,embedding_error=NULL,embedding_retry_at=0,ram_data=NULL,ram_model=NULL,ram_sha256=NULL,ram_error=NULL,ram_retry_at=0,ram_manually_edited=0,updated=? WHERE id=?""", (time.time(), image_id))
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

    def log_events(self, limit=300, category='all', query=''):
        return read_events(self.log_path, limit, category, query)

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
        db.create_function('image_name', 1, lambda path: str(path).rsplit('/', 1)[-1].casefold(), deterministic=True)
        try:
            with db:
                yield db
        finally:
            db.close()

    def permitted(self, path):
        path = Path(path).resolve()
        if self.excluded_path(path):
            raise ValueError('Image is in an excluded hidden folder')
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
            if 'people' not in result:
                raise ValueError('Model response must contain a people array (empty if no people are visible)')
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

    def order_candidates(self, candidates, sort_by, direction, interrupted):
        ordered, unavailable, fallbacks = [], [], 0
        for path in candidates:
            if interrupted():
                return []
            tie = (path.name.casefold(), str(path))
            try:
                if sort_by == 'size':
                    primary = path.stat().st_size
                elif sort_by == 'created':
                    primary, fallback = creation_time_ns(path, path.stat())
                    fallbacks += int(fallback)
                else:
                    primary = path.name.casefold()
                ordered.append(((primary, *tie), path))
            except OSError as exc:
                self.log.warning('Cannot read sort metadata for %s: %s', path, exc)
                unavailable.append(path)
        if fallbacks:
            self.log.info('Creation date unavailable for %s images; using modification time', fallbacks)
        ordered.sort(key=lambda item: item[0], reverse=direction == 'desc')
        return [path for _, path in ordered] + sorted(unavailable, key=str)

    def scan(self):
        if self.paused or not self.scan_lock.acquire(blocking=False):
            return
        with self.control_lock:
            scan_generation = self.generation
            roots = list(self.roots)
            backend = self.config.get('analysis_backend', 'llama')
            sort_by = self.config.get('scan_sort_by', 'name')
            sort_direction = self.config.get('scan_sort_direction', 'asc')
        with self.state_lock:
            self.reset_analysis_timing()
            self.state.update(running=True, roots={}, progress={
                'phase': 'discovering', 'scanned': 0, 'total': None, 'discovered': 0})
        self.log.info('Scan started')
        def interrupted():
            return self.stop.is_set() or self.paused or scan_generation != self.generation
        try:
            candidates = []
            root_errors = {}
            for root in roots:
                if interrupted():
                    return
                if self.excluded_directory(root):
                    self.log.info('Skipping hidden scan root: %s', root)
                    with self.state_lock:
                        self.state['roots'][str(root)] = 'Excluded hidden folder'
                    continue
                if not root.is_dir():
                    self.log.warning('Root unavailable: %s', root)
                    with self.state_lock:
                        self.state['roots'][str(root)] = 'Unavailable; index retained'
                    continue
                self.log.info('Scanning root: %s', root)
                errors = []
                root_errors[str(root)] = errors
                with self.state_lock:
                    self.state['roots'][str(root)] = 'Discovering images'
                for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
                    if interrupted():
                        return
                    dirs[:] = [d for d in dirs if not d.startswith('.') and not (Path(directory) / d).is_symlink()]
                    for name in files:
                        candidate = Path(directory) / name
                        if candidate.suffix.lower() not in EXTENSIONS or candidate.is_symlink():
                            continue
                        candidates.append(candidate)
                    with self.state_lock:
                        if scan_generation == self.generation:
                            self.state['progress']['discovered'] = len(candidates)
                for error in errors:
                    self.log.warning('Folder traversal failed: %s', error)
            # Overlapping configured roots should not count the same path twice.
            candidates = self.order_candidates(list(dict.fromkeys(candidates)), sort_by, sort_direction, interrupted)
            if interrupted():
                return
            self.log.info('Scan order: %s %s', sort_by, sort_direction)
            with self.state_lock:
                if scan_generation == self.generation:
                    self.state['progress'].update(phase='scanning', total=len(candidates), discovered=len(candidates))
            self.log.info('Discovered %s images to check', len(candidates))
            if backend in ('openclip', 'hybrid'):
                with self.state_lock:
                    self.state['progress'].update(phase='embedding', scanned=0)
                self.log.info('OpenCLIP and metadata pass started for %s images', len(candidates))
                for path in candidates:
                    if interrupted():
                        return
                    with self.state_lock:
                        self.state['current'] = str(path)
                    try:
                        self.process_embedding(self.permitted(path), scan_generation)
                    except OpenClipError:
                        raise
                    except Exception:
                        self.log.exception('Cannot embed %s', path)
                    with self.state_lock:
                        if scan_generation == self.generation:
                            self.state['progress']['scanned'] += 1
                with self.state_lock:
                    self.reset_analysis_timing()
                    self.state['progress'].update(phase='scanning', scanned=0)
                self.log.info('OpenCLIP and metadata pass finished')
            if self.config.get('recognize_enabled', False):
                with self.state_lock:
                    self.reset_analysis_timing()
                    self.state['progress'].update(phase='recognizing', scanned=0)
                self.log.info('RAM++ tagging pass started for %s images', len(candidates))
                for path in candidates:
                    if interrupted():
                        return
                    with self.state_lock:
                        self.state['current'] = str(path)
                    try:
                        self.process_recognition(self.permitted(path), scan_generation)
                    except RecognitionError:
                        raise
                    except Exception:
                        self.log.exception('Cannot tag %s with RAM++', path)
                    with self.state_lock:
                        if scan_generation == self.generation:
                            self.state['progress']['scanned'] += 1
                self.log.info('RAM++ tagging pass finished')
            if backend in ('llama', 'hybrid'):
                with self.state_lock:
                    self.reset_analysis_timing()
                    self.state['progress'].update(phase='scanning', scanned=0)
                self.log.info('Llama-server pass started')
            for candidate in candidates if backend in ('llama', 'hybrid') else []:
                if interrupted():
                    return
                with self.state_lock:
                    self.state['current'] = str(candidate)
                try:
                    self.process(self.permitted(candidate), scan_generation)
                except (OpenClipError, RecognitionError):
                    raise
                except Exception:
                    self.log.exception('Cannot index %s', candidate)
                finally:
                    with self.state_lock:
                        if scan_generation == self.generation:
                            self.state['progress']['scanned'] += 1
            with self.state_lock:
                if scan_generation == self.generation:
                    for root, errors in root_errors.items():
                        self.state['roots'][root] = '; '.join(errors) if errors else 'Scanned'
        finally:
            with self.state_lock:
                self.state.update(running=False, current=None, last_scan=time.time())
                if scan_generation == self.generation:
                    self.state['progress']['phase'] = 'paused' if self.paused else 'interrupted' if interrupted() else 'complete'
                    if self.state['progress']['total'] is not None and self.state['progress']['scanned'] < self.state['progress']['total'] and not interrupted():
                        self.state['progress']['phase'] = 'error'
            self.scan_lock.release()
            self.log.info('Scan finished%s', ' (paused)' if self.paused else '')

    @staticmethod
    def hash_file(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def search_text(path, result, metadata=None, ram_data=None):
        people = []
        for person in result.get('people', []):
            people.extend([person['position'], person['age_group'], person['gender_presentation'],
                           *person['clothing'], *person['accessories']])
        return ' '.join([path.name, result['description'], *result['scenes'],
                         *result['objects'], *result['colours'], *people,
                         metadata['raw'] if metadata else '', *(ram_data or {}).get('tags', []),
                         *(ram_data or {}).get('tags_chinese', [])]).lower()

    @staticmethod
    def ram_for_hash(row, digest):
        return json.loads(row['ram_data']) if row and row['ram_data'] and row['ram_sha256'] == digest else None

    def process_embedding(self, path, expected_generation=None):
        if self.excluded_path(path):
            return
        with self.control_lock:
            if self.paused or (expected_generation is not None and expected_generation != self.generation):
                return
            generation = self.generation
            config = dict(self.config)
            key = model_key(config)
            stat = path.stat()
            with self.connect() as db:
                row = db.execute('SELECT * FROM images WHERE path=?', (str(path),)).fetchone()
                if row and row['status'] == 'removed':
                    return
                unchanged = row and (row['size'], row['mtime_ns']) == (stat.st_size, stat.st_mtime_ns)
                if unchanged and row['image_embedding'] and row['embedding_model'] == key and row['embedding_sha256'] == row['sha256']:
                    return
                if unchanged and row['embedding_retry_at'] > time.time():
                    return
        digest = self.hash_file(path)
        metadata = read_a1111_metadata(path)
        metadata_json = json.dumps(metadata) if metadata else None
        with self.connect() as db:
            cached = db.execute("""SELECT * FROM images WHERE embedding_sha256=?
                AND embedding_model=? AND image_embedding IS NOT NULL AND status='done' ORDER BY analysis_version DESC,id LIMIT 1""", (digest, key)).fetchone()
        started = None
        try:
            if cached:
                vector = cached['image_embedding']
            else:
                with self.state_lock:
                    self.state['current'] = str(path)
                    if generation == self.generation:
                        started = self.active_analysis_started = time.monotonic()
                self.log.info('OpenCLIP embedding: %s', path)
                vector = self.encoder.image(path, config)
            _, palette, width, height = self.prepare(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError('File changed during OpenCLIP analysis; will retry')
            with self.control_lock:
                if self.paused or generation != self.generation:
                    return
                with self.connect() as db:
                    previous = db.execute('SELECT * FROM images WHERE path=?', (str(path),)).fetchone()
                    if previous and previous['status'] == 'removed':
                        return
                    caption_source = previous if previous and previous['analysis'] and (
                        previous['sha256'] == digest or (not previous['sha256'] and unchanged)) else cached
                    preserve = caption_source is not None and caption_source['analysis'] is not None
                    if preserve and caption_source['manually_edited']:
                        metadata = json.loads(caption_source['a1111_metadata']) if caption_source['a1111_metadata'] else None
                        metadata_json = json.dumps(metadata) if metadata else None
                    result = json.loads(caption_source['analysis']) if preserve else {
                        'description': '', 'scenes': [], 'objects': [], 'colours': [], 'people': []}
                    db.execute("""INSERT INTO images(path,size,mtime_ns,sha256,status,analysis,search_text,
                        palette,width,height,model,updated,a1111_metadata,analysis_version,
                        image_embedding,embedding_model,embedding_sha256,manually_edited)
                        VALUES(?,?,?,?,'done',?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,
                        sha256=excluded.sha256,status='done',analysis=excluded.analysis,search_text=excluded.search_text,
                        palette=excluded.palette,width=excluded.width,height=excluded.height,model=excluded.model,
                        updated=excluded.updated,a1111_metadata=excluded.a1111_metadata,analysis_version=excluded.analysis_version,manually_edited=excluded.manually_edited,
                        image_embedding=excluded.image_embedding,embedding_model=excluded.embedding_model,
                        embedding_sha256=excluded.embedding_sha256,embedding_error=NULL,embedding_retry_at=0,manually_edited=excluded.manually_edited""",
                        (str(path), stat.st_size, stat.st_mtime_ns, digest, json.dumps(result),
                         self.search_text(path, result, metadata, self.ram_for_hash(previous, digest)), json.dumps(palette), width, height,
                         caption_source['model'] if preserve else None, time.time(), metadata_json,
                         caption_source['analysis_version'] if preserve else 0, vector, key, digest,
                         caption_source['manually_edited'] if preserve else 0))
                    self.mark_indexed(db, path, stat)
            self.log.info('OpenCLIP %s: %s', 'hash reused' if cached else 'indexed', path)
        except Exception as exc:
            self.log.exception('OpenCLIP failed for %s', path)
            with self.control_lock:
                if generation == self.generation:
                    with self.connect() as db:
                        db.execute("""INSERT INTO images(path,size,mtime_ns,status,embedding_error,embedding_retry_at,updated)
                            VALUES(?,?,?,'error',?,?,?) ON CONFLICT(path) DO UPDATE SET
                            embedding_error=excluded.embedding_error,embedding_retry_at=excluded.embedding_retry_at,updated=excluded.updated
                            WHERE images.status!='removed'""",
                            (str(path), stat.st_size, stat.st_mtime_ns, str(exc)[:1000], time.time()+60, time.time()))
            if isinstance(exc, OpenClipError):
                raise
        finally:
            if started is not None:
                with self.state_lock:
                    if generation == self.generation:
                        self.analysis_count += 1
                        self.analysis_seconds += max(0, time.monotonic() - started)
                        self.active_analysis_started = None

    def process_recognition(self, path, expected_generation=None):
        if self.excluded_path(path):
            return
        with self.control_lock:
            if self.paused or (expected_generation is not None and expected_generation != self.generation):
                return
            generation, config = self.generation, dict(self.config)
            key, stat = recognition_key(config), path.stat()
            with self.connect() as db:
                row = db.execute('SELECT * FROM images WHERE path=?', (str(path),)).fetchone()
                if row and row['status'] == 'removed':
                    return
                unchanged = row and (row['size'], row['mtime_ns']) == (stat.st_size, stat.st_mtime_ns)
                if unchanged and row['ram_data'] and row['ram_sha256'] == row['sha256'] and (
                        row['ram_model'] == key or row['ram_manually_edited']):
                    return
                if unchanged and row['ram_retry_at'] > time.time():
                    return
        digest = row['sha256'] if unchanged and row['sha256'] else self.hash_file(path)
        metadata = (json.loads(row['a1111_metadata']) if unchanged and row['a1111_metadata']
                    else read_a1111_metadata(path))
        with self.connect() as db:
            cached = db.execute("""SELECT * FROM images WHERE ram_sha256=? AND ram_data IS NOT NULL
                AND status='done' AND (ram_model=? OR ram_manually_edited=1)
                ORDER BY ram_manually_edited DESC,id LIMIT 1""", (digest, key)).fetchone()
        started = None
        try:
            if cached:
                data = json.loads(cached['ram_data'])
            else:
                with self.state_lock:
                    if generation == self.generation:
                        started = self.active_analysis_started = time.monotonic()
                self.log.info('RAM++ tagging: %s', path)
                data = self.recognizer.image(path, config)
            validate_recognition_data(data)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError('File changed during RAM++ analysis; will retry')
            if unchanged and row['palette']:
                palette, width, height = json.loads(row['palette']), row['width'], row['height']
            else:
                _, palette, width, height = self.prepare(path)
            with self.control_lock:
                if self.paused or generation != self.generation:
                    return
                with self.connect() as db:
                    previous = db.execute('SELECT * FROM images WHERE path=?', (str(path),)).fetchone()
                    if previous and (previous['status'] == 'removed' or (
                            previous['ram_manually_edited'] and previous['ram_sha256'] == digest)):
                        return
                    preserve = previous and previous['analysis'] and (
                        previous['sha256'] == digest or (not previous['sha256'] and unchanged))
                    result = json.loads(previous['analysis']) if preserve else {
                        'description': '', 'scenes': [], 'objects': [], 'colours': [], 'people': []}
                    if preserve and previous['manually_edited']:
                        metadata = json.loads(previous['a1111_metadata']) if previous['a1111_metadata'] else None
                    db.execute("""INSERT INTO images(path,size,mtime_ns,sha256,status,analysis,search_text,
                        palette,width,height,model,updated,a1111_metadata,analysis_version,manually_edited,
                        ram_data,ram_model,ram_sha256,ram_manually_edited)
                        VALUES(?,?,?,?,'done',?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,
                        sha256=excluded.sha256,status='done',analysis=excluded.analysis,search_text=excluded.search_text,
                        palette=excluded.palette,width=excluded.width,height=excluded.height,model=excluded.model,
                        updated=excluded.updated,a1111_metadata=excluded.a1111_metadata,analysis_version=excluded.analysis_version,
                        manually_edited=excluded.manually_edited,ram_data=excluded.ram_data,ram_model=excluded.ram_model,
                        ram_sha256=excluded.ram_sha256,ram_manually_edited=excluded.ram_manually_edited,
                        ram_error=NULL,ram_retry_at=0""",
                        (str(path), stat.st_size, stat.st_mtime_ns, digest, json.dumps(result),
                         self.search_text(path, result, metadata, data), json.dumps(palette), width, height,
                         previous['model'] if preserve else None, time.time(), json.dumps(metadata) if metadata else None,
                         previous['analysis_version'] if preserve else 0, previous['manually_edited'] if preserve else 0,
                         json.dumps(data), cached['ram_model'] if cached else key, digest,
                         cached['ram_manually_edited'] if cached else 0))
                    self.mark_indexed(db, path, stat)
            self.log.info('RAM++ %s: %s', 'hash reused' if cached else 'indexed', path)
        except Exception as exc:
            self.log.exception('RAM++ failed for %s', path)
            with self.control_lock:
                if generation == self.generation:
                    with self.connect() as db:
                        db.execute("""INSERT INTO images(path,size,mtime_ns,status,ram_error,ram_retry_at,updated)
                            VALUES(?,?,?,'error',?,?,?) ON CONFLICT(path) DO UPDATE SET
                            ram_error=excluded.ram_error,ram_retry_at=excluded.ram_retry_at,updated=excluded.updated
                            WHERE images.status!='removed'""",
                            (str(path), stat.st_size, stat.st_mtime_ns, str(exc)[:1000], time.time()+60, time.time()))
            if isinstance(exc, RecognitionError):
                raise
        finally:
            if started is not None:
                with self.state_lock:
                    if generation == self.generation:
                        self.analysis_count += 1
                        self.analysis_seconds += max(0, time.monotonic() - started)
                        self.active_analysis_started = None

    def process(self, path, expected_generation=None):
        backend = self.config.get('analysis_backend', 'llama')
        if backend in ('openclip', 'hybrid'):
            self.process_embedding(path, expected_generation)
        if self.config.get('recognize_enabled', False):
            self.process_recognition(path, expected_generation)
        if backend in ('llama', 'hybrid'):
            self._process_llama(path, expected_generation)

    def _process_llama(self, path, expected_generation=None):
        if self.excluded_path(path):
            return
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
                if unchanged and row['manually_edited']:
                    return
                if unchanged and row['sha256'] and ((row['status'] == 'done' and row['analysis_version'] >= ANALYSIS_VERSION) or row['retry_at'] > time.time()):
                    return
        # Stream files without holding the control lock while reading network storage.
        digest = self.hash_file(path)
        try:
            metadata = read_a1111_metadata(path)
        except (OSError, ValueError, SyntaxError) as exc:
            self.log.warning('Could not read A1111 metadata for %s: %s', path, exc)
            metadata = None
        metadata_json = json.dumps(metadata) if metadata else None
        after_hash = path.stat()
        if (after_hash.st_size, after_hash.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('File changed while hashing; will retry on next scan')
        with self.control_lock:
            if self.paused or generation != self.generation:
                return
            with self.connect() as db:
                row = db.execute('SELECT * FROM images WHERE path=?', (str(path),)).fetchone()
                if row and row['status'] == 'removed':
                    return
                unchanged = row and (row['size'], row['mtime_ns']) == (stat.st_size, stat.st_mtime_ns)
                if row and row['manually_edited'] and row['sha256'] == digest:
                    db.execute('UPDATE images SET size=?,mtime_ns=? WHERE id=?', (stat.st_size,stat.st_mtime_ns,row['id']))
                    return
                if unchanged and row['status'] == 'done' and row['analysis_version'] >= ANALYSIS_VERSION:
                    # Upgrade old records without calling the model again.
                    db.execute('UPDATE images SET sha256=? WHERE id=?', (digest, row['id']))
                    return
                cached = db.execute("SELECT * FROM images WHERE sha256=? AND status='done' AND analysis_version>=? ORDER BY id LIMIT 1",
                                    (digest, ANALYSIS_VERSION)).fetchone()
                if cached:
                    if cached['manually_edited']:
                        metadata = json.loads(cached['a1111_metadata']) if cached['a1111_metadata'] else None
                        metadata_json = json.dumps(metadata) if metadata else None
                    result = json.loads(cached['analysis'])
                    db.execute("""INSERT INTO images(path,size,mtime_ns,sha256,status,analysis,search_text,
                        palette,width,height,model,updated,a1111_metadata,analysis_version,manually_edited) VALUES(?,?,?,?,'done',?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,
                        sha256=excluded.sha256,status='done',analysis=excluded.analysis,
                        search_text=excluded.search_text,palette=excluded.palette,width=excluded.width,
                        height=excluded.height,model=excluded.model,error=NULL,attempts=0,retry_at=0,
                        updated=excluded.updated,a1111_metadata=excluded.a1111_metadata,analysis_version=excluded.analysis_version,manually_edited=excluded.manually_edited""",
                        (str(path), stat.st_size, stat.st_mtime_ns, digest, cached['analysis'],
                         self.search_text(path, result, metadata, self.ram_for_hash(row, digest)), cached['palette'], cached['width'],
                         cached['height'], cached['model'], time.time(), metadata_json, ANALYSIS_VERSION,cached['manually_edited']))
                    self.mark_indexed(db, path, stat)
                    self.log.info('Reused analysis by SHA-256: %s (from %s)', path, cached['path'])
                    return
                if unchanged and row['retry_at'] > time.time():
                    return
                attempts = row['attempts'] if unchanged else 0
                db.execute("""INSERT INTO images(path,size,mtime_ns,sha256,a1111_metadata) VALUES(?,?,?,?,?)
                    ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,
                    sha256=excluded.sha256,a1111_metadata=excluded.a1111_metadata,manually_edited=0,
                    status=CASE WHEN (images.image_embedding IS NOT NULL AND images.embedding_sha256=excluded.sha256 OR images.ram_data IS NOT NULL AND images.ram_sha256=excluded.sha256) THEN 'done' ELSE 'pending' END,
                    analysis=CASE WHEN (images.image_embedding IS NOT NULL AND images.embedding_sha256=excluded.sha256 OR images.ram_data IS NOT NULL AND images.ram_sha256=excluded.sha256) THEN images.analysis ELSE NULL END,
                    search_text=CASE WHEN (images.image_embedding IS NOT NULL AND images.embedding_sha256=excluded.sha256 OR images.ram_data IS NOT NULL AND images.ram_sha256=excluded.sha256) THEN images.search_text ELSE NULL END,
                    palette=CASE WHEN (images.image_embedding IS NOT NULL AND images.embedding_sha256=excluded.sha256 OR images.ram_data IS NOT NULL AND images.ram_sha256=excluded.sha256) THEN images.palette ELSE NULL END,error=NULL""",
                    (str(path), stat.st_size, stat.st_mtime_ns, digest, metadata_json))
        with self.state_lock:
            self.state['current'] = str(path)
        self.log.info('Analysing image: %s', path)
        started = time.monotonic()
        with self.state_lock:
            if generation == self.generation:
                self.active_analysis_started = started
        try:
            result, palette, width, height = self.analyse(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError('File changed during analysis; will retry')
            search = self.search_text(path, result, metadata, self.ram_for_hash(row, digest))
            with self.control_lock:
                if generation != self.generation:
                    self.log.info('Discarded analysis after index or settings changed: %s', path)
                    return
                with self.connect() as db:
                    updated = db.execute('''UPDATE images SET status='done',analysis=?,search_text=?,palette=?,
                        width=?,height=?,model=?,error=NULL,attempts=0,retry_at=0,updated=?,a1111_metadata=?,analysis_version=? WHERE path=? AND status!='removed' AND manually_edited=0 ''',
                        (json.dumps(result), search, json.dumps(palette), width, height,
                         model, time.time(), metadata_json, ANALYSIS_VERSION, str(path)))
                    if not updated.rowcount:
                        self.log.info('Discarded analysis for removed image: %s', path)
                        return
                    self.mark_indexed(db, path, stat)
            self.log.info('Indexed image in %.1fs: %s', time.monotonic() - started, path)
        except Exception as exc:
            self.log.exception('Analysis failed for %s', path)
            with self.control_lock:
                if generation != self.generation:
                    return
                with self.connect() as db:
                    db.execute("UPDATE images SET status=CASE WHEN (image_embedding IS NOT NULL AND embedding_sha256=sha256 OR ram_data IS NOT NULL AND ram_sha256=sha256) AND analysis IS NOT NULL THEN 'done' ELSE 'error' END,error=?,attempts=?,retry_at=?,updated=? WHERE path=? AND status!='removed' AND manually_edited=0",
                        (str(exc)[:1000], attempts + 1, time.time() + min(3600, 60 * 2 ** min(attempts, 6)), time.time(), str(path)))

        finally:
            with self.state_lock:
                if generation == self.generation:
                    self.analysis_count += 1
                    self.analysis_seconds += max(0, time.monotonic() - started)
                    self.active_analysis_started = None

    def run(self):
        while not self.stop.is_set():
            self.wake.clear()
            try:
                self.scan()
            except Exception:
                self.log.exception('Scanner failed; will retry on next scan')
            self.wake.wait(self.config.get('scan_interval_seconds', 3600))

    def search(self, q='', scene='', object_name='', colour='', offset=0, limit=60, mode='keyword',
               sort_by='relevance', direction='desc'):
        if mode not in ('keyword', 'semantic', 'hybrid'):
            raise ValueError('Unknown search mode')
        if sort_by not in ('updated', 'created', 'name', 'relevance') or direction not in ('asc', 'desc'):
            raise ValueError('Unknown result sort order')
        if sort_by == 'created':
            self.ensure_creation_dates()
        sql_direction = 'ASC' if direction == 'asc' else 'DESC'
        order = {
            'updated': f'last_indexed_at IS NULL, last_indexed_at {sql_direction}, id {sql_direction}',
            'created': f'created_at_ns IS NULL, created_at_ns {sql_direction}, image_name(path) {sql_direction}, id {sql_direction}',
            'name': f'image_name(path) {sql_direction}, path {sql_direction}, id {sql_direction}',
            'relevance': 'last_indexed_at IS NULL, last_indexed_at DESC, id DESC',
        }[sort_by]
        with self.control_lock:
            config = dict(self.config)
        enabled = config.get('analysis_backend', 'llama') in ('openclip', 'hybrid')
        if mode == 'semantic' and not enabled:
            raise OpenClipError('Select OpenCLIP or Both in Settings to enable semantic search')
        if not enabled or not q.strip():
            mode = 'keyword'
        clauses, params = ["status='done'"], []
        for field, value in [('scenes', scene), ('objects', object_name), ('colours', colour)]:
            if value.strip():
                clauses.append(f"EXISTS (SELECT 1 FROM json_each(images.analysis, '$.{field}') WHERE instr(lower(value), ?) > 0)")
                params.append(value.strip().lower())
        terms = q.lower().split()
        if mode == 'keyword':
            for term in terms:
                clauses.append('instr(search_text, ?) > 0')
                params.append(term)
        where = ' AND '.join(clauses)
        semantic_scores = {}
        with self.connect() as db:
            if mode == 'keyword':
                total = db.execute('SELECT count(*) FROM images WHERE ' + where, params).fetchone()[0]
                rows = db.execute('SELECT * FROM images WHERE ' + where + ' ORDER BY ' + order + ' LIMIT ? OFFSET ?',
                                  [*params, limit, offset]).fetchall()
            else:
                candidates = db.execute('SELECT * FROM images WHERE ' + where + ' ORDER BY ' + order, params).fetchall()
                vectors = [r for r in candidates if r['image_embedding'] and r['embedding_model'] == model_key(config)
                           and r['embedding_sha256'] == r['sha256']]
                ranked = rank_vectors(vectors, self.encoder.text(q, config)) if vectors else []
                semantic_scores = dict(ranked)
                lexical = [r['id'] for r in candidates if all(term in (r['search_text'] or '') for term in terms)]
                if mode == 'semantic':
                    ids = [image_id for image_id, _ in ranked]
                else:
                    # Reciprocal rank fusion keeps exact metadata matches useful alongside visual similarity.
                    fused = {}
                    for ranking in (lexical, [image_id for image_id, _ in ranked]):
                        for rank, image_id in enumerate(ranking, 1):
                            fused[image_id] = fused.get(image_id, 0) + 1 / (60 + rank)
                    ids = sorted(fused, key=lambda image_id: (-fused[image_id], image_id))
                total = len(ids)
                by_id = {r['id']: r for r in candidates}
                if sort_by != 'relevance':
                    eligible = set(ids)
                    ids = [row['id'] for row in candidates if row['id'] in eligible]
                elif direction == 'asc':
                    ids.reverse()
                rows = [by_id[image_id] for image_id in ids[offset:offset+limit]]
        return {'total': total, 'search_mode': mode, 'sort_by': sort_by, 'direction': direction, 'items': [dict(
            id=r['id'], path=r['path'], last_indexed_at=r['last_indexed_at'],
            created_at=r['created_at_ns']/1_000_000_000 if r['created_at_ns'] is not None else None,
            creation_date_fallback=bool(r['creation_date_fallback']) if r['creation_date_fallback'] is not None else None, width=r['width'], height=r['height'],
            analysis=json.loads(r['analysis']), palette=json.loads(r['palette']),
            a1111_metadata=json.loads(r['a1111_metadata']) if r['a1111_metadata'] else None,
            embedded=bool(r['image_embedding'] and r['embedding_model'] == model_key(config)
                          and r['embedding_sha256'] == r['sha256']),
            semantic_score=semantic_scores.get(r['id']), manually_edited=bool(r['manually_edited']),
            ram_data=self.ram_for_hash(r, r['sha256']), ram_manually_edited=bool(r['ram_manually_edited'])) for r in rows]}
