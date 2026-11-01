import argparse
from contextlib import asynccontextmanager
import io
import json
import logging
import sqlite3
import struct
import math
from pathlib import Path
import threading
import tempfile
import os
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps
import uvicorn

from .core import Index
from .settings import RuntimeSettings
from .embeddings import OpenClipError, model_key
from .runner import RESTART_EXIT_CODE, supervise


class ImageEditRequest(BaseModel):
    description: str = Field(max_length=8000)
    scenes: list[str] = Field(max_length=100)
    objects: list[str] = Field(max_length=100)
    colours: list[str] = Field(max_length=100)
    people: list[dict] = Field(default_factory=list, max_length=10)
    a1111_raw: str = Field(default='', max_length=128000)


class ClearRequest(BaseModel):
    confirm: Literal[True]


def create_app(config, restart_callback=None):
    index = Index(config)

    @asynccontextmanager
    async def lifespan(app):
        worker = threading.Thread(target=index.run, daemon=True)
        worker.start()
        yield
        index.stop.set()
        index.wake.set()
        worker.join(timeout=2)

    app = FastAPI(title='Image Analyser', lifespan=lifespan)
    app.state.index = index
    app.state.instance_id = uuid.uuid4().hex
    app.state.restarting = False
    restart_lock = threading.Lock()

    @app.get('/', response_class=HTMLResponse)
    def home():
        return Path(__file__).with_name('index.html').read_text()

    @app.get('/api/database/backup')
    def download_backup():
        path = index.backup_database()
        return FileResponse(path, media_type='application/vnd.sqlite3', filename='imageanalyser-backup.sqlite3',
                            background=BackgroundTask(path.unlink, missing_ok=True))

    @app.post('/api/database/restore')
    async def restore_backup(request: Request, confirm: bool = Query(False)):
        if not confirm:
            raise HTTPException(400, 'Confirm database restore before uploading')
        fd, filename = tempfile.mkstemp(suffix='.sqlite3')
        os.close(fd)
        path = Path(filename)
        try:
            size = 0
            with path.open('wb') as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 256 * 1024 * 1024:
                        raise HTTPException(413, 'Backup exceeds the 256 MB upload limit')
                    stream.write(chunk)
            if not size:
                raise HTTPException(400, 'Select a SQLite backup file')
            try:
                safety = await run_in_threadpool(index.restore_database, path)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            except (OSError, sqlite3.Error):
                index.log.exception('Database restore failed')
                raise HTTPException(500, 'Restore failed; see Logs for details')
            return {'restored': True, 'paused': True, 'safety_backup': safety}
        finally:
            path.unlink(missing_ok=True)

    @app.post('/api/backend/restart', status_code=202)
    def restart_backend(body: ClearRequest):
        if restart_callback is None:
            raise HTTPException(503, 'Restart requires launching with the imageanalyser command')
        with restart_lock:
            if app.state.restarting:
                return {'restarting': True, 'instance_id': app.state.instance_id}
            app.state.restarting = True

        def restart():
            index.log.info('Backend restart requested by user')
            with index.control_lock:
                index.generation += 1
                index.stop.set()
                index.wake.set()
            restart_callback()

        return JSONResponse({'restarting': True, 'instance_id': app.state.instance_id}, status_code=202,
                            background=BackgroundTask(restart))

    @app.get('/api/settings')
    def get_settings():
        return index.get_settings()

    @app.put('/api/settings')
    def save_settings(body: RuntimeSettings):
        try:
            return index.save_settings(body)
        except (OSError, sqlite3.Error):
            index.log.exception('Failed to save settings')
            raise HTTPException(500, 'Could not save settings; see Logs for details')

    @app.get('/api/search')
    def search(q: str = Query('', max_length=500), scene: str = '', object: str = '', colour: str = '',
               offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=100),
               mode: Literal['keyword', 'semantic', 'hybrid'] = 'keyword'):
        try:
            return index.search(q, scene, object, colour, offset, limit, mode)
        except OpenClipError as exc:
            index.log.exception('OpenCLIP search failed')
            raise HTTPException(503, str(exc))

    @app.get('/api/status')
    def status():
        with index.connect() as db:
            counts = dict(db.execute("SELECT status,count(*) FROM images WHERE status!='removed' GROUP BY status").fetchall())
            embedded = db.execute("""SELECT count(*) FROM images WHERE status='done' AND image_embedding IS NOT NULL
                AND embedding_model=? AND embedding_sha256=sha256""", (model_key(index.config),)).fetchone()[0]
            cleared = db.execute("SELECT value FROM settings WHERE key='diagnostics_cleared_at'").fetchone()
            since = float(cleared['value']) if cleared else None
            errors = [dict(r) for r in db.execute(
                "SELECT path,coalesce(embedding_error,error) AS error FROM images WHERE (status='error' OR embedding_error IS NOT NULL OR error IS NOT NULL) AND (? IS NULL OR updated>?) ORDER BY id DESC LIMIT 10",
                (since, since))]
        with index.state_lock:
            return {**index.state, 'instance_id': app.state.instance_id, 'restart_supported': restart_callback is not None, 'restarting': app.state.restarting, 'paused': index.paused, 'roots': dict(index.state['roots']), 'progress': index.progress_snapshot(), 'counts': counts, 'embedding_count': embedded, 'analysis_backend': index.config.get('analysis_backend', 'llama'), 'errors': errors}

    @app.post('/api/scanner/details/clear')
    def clear_scanner_details(body: ClearRequest):
        index.clear_scanner_details()
        return {'cleared': True}

    @app.post('/api/images/clear')
    def clear_images(body: ClearRequest):
        return {'cleared': index.clear(), 'paused': True}

    @app.post('/api/scan', status_code=202)
    def scan_now(body: ClearRequest):
        return {'status': index.request_scan()}

    @app.post('/api/scan/resume')
    def resume_scan(body: ClearRequest):
        index.resume()
        return {'paused': False}

    @app.get('/api/logs')
    def logs(limit: int = Query(300, ge=1, le=1000),
             category: Literal['all', 'problems', 'error', 'warning', 'success', 'activity'] = 'all',
             q: str = Query('', max_length=500)):
        return index.log_events(limit, category, q)

    @app.put('/api/images/{image_id}')
    def edit_image(image_id: int, body: ImageEditRequest):
        try:
            saved = index.edit_image(image_id, body.model_dump(exclude={'a1111_raw'}), body.a1111_raw)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        if not saved:
            raise HTTPException(404, 'Image not found in index')
        return {'saved': True}

    @app.delete('/api/images/{image_id}')
    def remove_image(image_id: int, body: ClearRequest):
        if not index.remove_image(image_id):
            raise HTTPException(404, 'Image not found in index')
        return {'removed': True}

    @app.get('/api/images/{image_id}/openclip')
    def openclip_details(image_id: int):
        with index.connect() as db:
            row = db.execute("SELECT * FROM images WHERE id=? AND status!='removed'", (image_id,)).fetchone()
        if not row or not row['image_embedding']:
            raise HTTPException(404, 'No OpenCLIP embedding is stored for this image')
        raw = row['image_embedding']
        if not isinstance(raw, bytes) or len(raw) % 4 or len(raw) > 65536:
            raise HTTPException(409, 'The stored OpenCLIP vector is invalid; rescan this image')
        vector = list(struct.unpack('<' + 'f' * (len(raw) // 4), raw))
        if any(not math.isfinite(value) for value in vector):
            raise HTTPException(409, 'The stored OpenCLIP vector is invalid; rescan this image')
        architecture = weights = None
        try:
            stored = json.loads((row['embedding_model'] or '').removeprefix('openclip-v1:'))
            if isinstance(stored, list) and len(stored) == 2:
                architecture, weights = stored
        except (ValueError, TypeError):
            pass
        return {'image_id': image_id, 'model': architecture, 'pretrained_weights': weights,
                'model_key': row['embedding_model'], 'dimensions': len(vector),
                'format': 'normalized float32, little-endian', 'sha256': row['embedding_sha256'],
                'matches_file_hash': row['embedding_sha256'] == row['sha256'],
                'matches_current_model': row['embedding_model'] == model_key(index.config),
                'image_sha256': row['sha256'], 'last_error': row['embedding_error'],
                'retry_at': row['embedding_retry_at'] or None, 'embedding': vector}

    @app.get('/api/images/{image_id}/thumbnail')
    def thumbnail(image_id: int):
        with index.connect() as db:
            row = db.execute("SELECT path FROM images WHERE id=? AND status!='removed'", (image_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Unknown image')
        try:
            path = index.permitted(row['path'])
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((480, 360))
                out = io.BytesIO()
                image.save(out, format='JPEG', quality=80)
                return Response(out.getvalue(), media_type='image/jpeg')
        except (OSError, ValueError):
            raise HTTPException(404, 'Image unavailable')

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config_path = Path(args.config)
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError:
        parser.error(
            f"Configuration file not found: {config_path}. "
            "Copy config.example.json to this path and edit roots, llama_url, and model before starting."
        )
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read configuration {config_path}: {exc}")
    if not args.worker:
        raise SystemExit(supervise(config_path.resolve()))

    restart_requested = threading.Event()

    def restart():
        restart_requested.set()
        server.should_exit = True

    app = create_app(config, restart_callback=restart)
    server = uvicorn.Server(uvicorn.Config(app, host=config.get('host', '127.0.0.1'),
                                         port=config.get('port', 8090), workers=1,
                                         timeout_graceful_shutdown=5))
    server.run()
    if restart_requested.is_set():
        raise SystemExit(RESTART_EXIT_CODE)


if __name__ == '__main__':
    main()
