from pathlib import Path
import struct
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from imageanalyser.app import create_app
from imageanalyser.core import Index
from imageanalyser.embeddings import OpenClipError, model_key


class ResultSortingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'photos'
        self.root.mkdir()
        self.paths = [self.root / name for name in ('zeta.png', 'Alpha.png', 'middle.png')]
        for path, colour in zip(self.paths, ('red', 'blue', 'green')):
            Image.new('RGB', (30, 30), colour).save(path)
        self.config = {'roots': [str(self.root)], 'data_dir': str(Path(self.temp.name) / 'data'),
                       'llama_url': 'http://localhost:8080/v1', 'model': 'test'}
        self.index = Index(self.config)
        self.addCleanup(self.index.log_handler.close)
        self.caption = {'description': 'A photo', 'scenes': [], 'objects': [], 'colours': [], 'people': []}
        def analyse(path):
            _, palette, width, height = self.index.prepare(path)
            return self.caption, palette, width, height
        with patch.object(self.index, 'analyse', side_effect=analyse):
            self.index.scan()
        with self.index.connect() as db:
            for path, created, tagged in zip(self.paths, (100, 300, 200), (200, 300, 100)):
                db.execute('UPDATE images SET created_at_ns=?,last_indexed_at=?,updated=1 WHERE path=?',
                           (created, tagged, str(path)))

    def names(self, **kwargs):
        return [Path(item['path']).name for item in self.index.search(**kwargs)['items']]

    def test_date_and_name_order_in_both_directions_and_pagination(self):
        orders = {'name': ['Alpha.png', 'middle.png', 'zeta.png'],
                  'created': ['zeta.png', 'middle.png', 'Alpha.png'],
                  'updated': ['middle.png', 'zeta.png', 'Alpha.png']}
        for sort_by, expected in orders.items():
            with self.subTest(sort_by=sort_by):
                self.assertEqual(self.names(sort_by=sort_by, direction='asc'), expected)
                self.assertEqual(self.names(sort_by=sort_by, direction='desc'), list(reversed(expected)))
                self.assertEqual(self.names(sort_by=sort_by, direction='asc', offset=1, limit=1), expected[1:2])
        self.assertEqual(self.names(), ['Alpha.png', 'zeta.png', 'middle.png'])

    def test_api_sort_validation_and_successful_tag_date_field(self):
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        response = client.get('/api/search', params={'sort_by': 'updated', 'direction': 'desc'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['items'][0]['last_indexed_at'], 300)
        self.assertEqual(client.get('/api/search?sort_by=invalid').status_code, 422)
        self.assertEqual(client.get('/api/search?direction=invalid').status_code, 422)
        with self.assertRaises(ValueError):
            self.index.search(sort_by='invalid')

    def test_creation_dates_are_backfilled_once_and_unknown_dates_sort_last(self):
        with self.index.connect() as db:
            db.execute('UPDATE images SET created_at_ns=NULL,creation_date_fallback=NULL')
        dates = dict(zip(self.paths, (100, 300, 200)))
        with patch('imageanalyser.core.creation_time_ns', side_effect=lambda path, stat: (dates[path], path == self.paths[0])) as birth:
            self.assertEqual(self.names(sort_by='created', direction='asc'), ['zeta.png', 'middle.png', 'Alpha.png'])
            self.names(sort_by='created', direction='desc')
            self.assertEqual(birth.call_count, 3)
        self.assertTrue(self.index.search(sort_by='created', direction='asc')['items'][0]['creation_date_fallback'])
        self.paths[0].unlink()
        with self.index.connect() as db:
            db.execute('UPDATE images SET created_at_ns=NULL WHERE path=?', (str(self.paths[0]),))
        for direction in ('asc', 'desc'):
            self.assertEqual(self.names(sort_by='created', direction=direction)[-1], 'zeta.png')

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'OpenCLIP extra required')
    def test_result_sort_overrides_semantic_ranking_before_pagination(self):
        self.index.config['analysis_backend'] = 'openclip'
        with self.index.connect() as db:
            for path, values in zip(self.paths, [(1, 0), (0.6, 0.8), (0, 1)]):
                db.execute('UPDATE images SET image_embedding=?,embedding_model=?,embedding_sha256=sha256 WHERE path=?',
                           (struct.pack('<ff', *values), model_key(self.index.config), str(path)))
        with patch.object(self.index.encoder, 'text', return_value=struct.pack('<ff', 1, 0)):
            self.assertEqual(self.names(q='photo', mode='semantic'), ['zeta.png', 'Alpha.png', 'middle.png'])
            self.assertEqual(self.names(q='photo', mode='semantic', sort_by='updated', direction='desc'),
                             ['Alpha.png', 'zeta.png', 'middle.png'])
            page = self.index.search('photo', mode='semantic', sort_by='updated', offset=1, limit=1)
            self.assertEqual(Path(page['items'][0]['path']).name, 'zeta.png')
            self.assertAlmostEqual(page['items'][0]['semantic_score'], 1)
            self.assertEqual(self.names(q='photo', mode='hybrid', sort_by='name', direction='asc'),
                             ['Alpha.png', 'middle.png', 'zeta.png'])

    def test_legacy_last_indexed_timestamp_migration(self):
        with self.index.connect() as db:
            db.execute('DROP INDEX images_last_indexed')
            db.execute('ALTER TABLE images DROP COLUMN last_indexed_at')
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        with restarted.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM images WHERE last_indexed_at=updated").fetchone()[0], 3)

    def test_failed_attempt_does_not_change_last_tagged_date(self):
        self.index.config['analysis_backend'] = 'openclip'
        with patch.object(self.index.encoder, 'image', side_effect=OpenClipError('Model offline')):
            with self.assertRaises(OpenClipError):
                self.index.process_embedding(self.paths[0])
        with self.index.connect() as db:
            row = db.execute('SELECT updated,last_indexed_at FROM images WHERE path=?', (str(self.paths[0]),)).fetchone()
            self.assertEqual(row['last_indexed_at'], 200)
            self.assertGreater(row['updated'], 200)

    def test_manual_tag_edit_updates_latest_tagging_date(self):
        item = self.index.search(q='zeta')['items'][0]
        with patch('imageanalyser.core.time.time', return_value=999):
            self.index.edit_image(item['id'], self.caption, '')
        self.assertEqual(self.names(sort_by='updated')[0], 'zeta.png')
        self.assertEqual(self.index.search(q='zeta')['items'][0]['last_indexed_at'], 999)
