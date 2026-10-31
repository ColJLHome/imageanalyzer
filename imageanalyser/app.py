import argparse
from contextlib import asynccontextmanager
import io
import json
import logging
import sqlite3
from pathlib import Path
import threading
from typing import Literal

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageOps
import uvicorn

from .core import Index
from .settings import RuntimeSettings


class ClearRequest(BaseModel):
    confirm: Literal[True]


def create_app(config):
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

    @app.get('/', response_class=HTMLResponse)
    def home():
        return Path(__file__).with_name('index.html').read_text()

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
               offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=100)):
        return index.search(q, scene, object, colour, offset, limit)

    @app.get('/api/status')
    def status():
        with index.connect() as db:
            counts = dict(db.execute("SELECT status,count(*) FROM images WHERE status!='removed' GROUP BY status").fetchall())
            cleared = db.execute("SELECT value FROM settings WHERE key='diagnostics_cleared_at'").fetchone()
            since = float(cleared['value']) if cleared else None
            errors = [dict(r) for r in db.execute(
                "SELECT path,error FROM images WHERE status='error' AND (? IS NULL OR updated>?) ORDER BY id DESC LIMIT 10",
                (since, since))]
        with index.state_lock:
            return {**index.state, 'paused': index.paused, 'roots': dict(index.state['roots']), 'counts': counts, 'errors': errors}

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
    def logs(limit: int = Query(300, ge=1, le=1000)):
        return {'lines': index.logs(limit)}

    @app.delete('/api/images/{image_id}')
    def remove_image(image_id: int, body: ClearRequest):
        if not index.remove_image(image_id):
            raise HTTPException(404, 'Image not found in index')
        return {'removed': True}

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
    uvicorn.run(create_app(config), host=config.get('host', '127.0.0.1'), port=config.get('port', 8090), workers=1)


if __name__ == '__main__':
    main()
