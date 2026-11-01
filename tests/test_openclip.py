import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from imageanalyser.core import Index
from imageanalyser.embeddings import OpenClipError, model_key


def vector(a, b):
    return struct.pack('<ff', a, b)


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def image(self, path, config):
        self.calls.append(path)
        with Image.open(path) as image:
            red, _, blue = image.convert('RGB').getpixel((0, 0))
        return vector(1, 0) if red > blue else vector(0, 1)

    def text(self, query, config):
        return vector(1, 0) if 'red' in query else vector(0, 1)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'Install the openclip extra for semantic search tests')
class OpenClipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'photos'
        self.root.mkdir()
        self.red, self.blue = self.root / 'red.png', self.root / 'blue.png'
        info = PngInfo()
        info.add_text('parameters', 'crimson jacket\nNegative prompt: blur\nSteps: 20, Seed: 42, Model: studio')
        Image.new('RGB', (40, 40), 'red').save(self.red, pnginfo=info)
        Image.new('RGB', (40, 40), 'blue').save(self.blue)
        self.config = {'roots': [str(self.root)], 'data_dir': str(self.base / 'data'),
                       'llama_url': 'http://localhost:8080/v1', 'model': 'test',
                       'analysis_backend': 'openclip'}
        self.index = Index(self.config)
        self.addCleanup(self.index.log_handler.close)
        self.encoder = self.index.encoder = FakeEncoder()
        self.caption = {'description': 'A detailed caption', 'scenes': ['studio'],
                        'objects': ['jacket'], 'colours': ['red'], 'people': []}

    def test_embedding_only_and_semantic_and_metadata_search(self):
        with patch.object(self.index, 'analyse') as llama:
            self.index.scan()
            llama.assert_not_called()
        self.assertEqual(len(self.encoder.calls), 2)
        self.assertEqual(self.index.search()['total'], 2)
        self.assertEqual(self.index.search('42')['total'], 1)
        ranked = self.index.search('a red garment', mode='semantic')
        self.assertEqual(ranked['items'][0]['path'], str(self.red))
        self.assertAlmostEqual(ranked['items'][0]['semantic_score'], 1)
        self.assertTrue(ranked['items'][0]['embedded'])
        combined = self.index.search('studio', mode='hybrid')
        self.assertEqual(combined['total'], 2)
        self.index.scan()
        self.assertEqual(len(self.encoder.calls), 2)

    def test_hybrid_embeds_every_image_before_any_caption(self):
        self.index.config['analysis_backend'] = 'hybrid'
        events = []
        original = self.encoder.image
        def image(path, config):
            events.append(('embed', path))
            return original(path, config)
        def caption(path, generation):
            events.append(('caption', path))
        with patch.object(self.encoder, 'image', side_effect=image), \
             patch.object(self.index, '_process_llama', side_effect=caption):
            self.index.scan()
        self.assertEqual([event[0] for event in events], ['embed', 'embed', 'caption', 'caption'])

    def test_preserves_existing_analysis_and_reuses_hash_after_move(self):
        self.index.config['analysis_backend'] = 'llama'
        def caption(path):
            _, palette, width, height = self.index.prepare(path)
            return self.caption, palette, width, height
        with patch.object(self.index, 'analyse', side_effect=caption):
            self.index.scan()
        self.index.config['analysis_backend'] = 'openclip'
        self.index.scan()
        self.assertEqual(self.index.search()['items'][0]['analysis'], self.caption)
        moved = self.root / 'moved.png'
        self.red.rename(moved)
        self.index.scan()
        self.assertEqual(len(self.encoder.calls), 2)
        item = self.index.search('moved')['items'][0]
        self.assertEqual(item['analysis'], self.caption)
        self.assertIn('crimson jacket', item['a1111_metadata']['raw'])

    def test_different_model_and_changed_file_require_new_embeddings(self):
        self.index.scan()
        self.index.config['openclip_pretrained'] = 'different-weights'
        self.assertEqual(self.index.search('red', mode='semantic')['total'], 0)
        self.index.scan()
        self.assertEqual(len(self.encoder.calls), 4)
        Image.new('RGB', (50, 50), 'blue').save(self.red)
        self.index.scan()
        self.assertEqual(len(self.encoder.calls), 5)
        with self.index.connect() as db:
            row = db.execute('SELECT * FROM images WHERE path=?', (str(self.red),)).fetchone()
            self.assertEqual(row['embedding_model'], model_key(self.index.config))
            self.assertEqual(row['embedding_sha256'], row['sha256'])

    def test_llama_failure_does_not_hide_embedding_or_metadata(self):
        self.index.config['analysis_backend'] = 'hybrid'
        with patch.object(self.index, 'analyse', side_effect=RuntimeError('Caption server offline')):
            self.index.scan()
        self.assertEqual(self.index.search('red', mode='semantic')['total'], 2)
        self.assertEqual(self.index.search('studio')['total'], 1)
        self.assertEqual(self.index.search()['total'], 2)

    def test_removal_and_clear_during_embedding_prevent_resurrection(self):
        self.index.scan()
        image_id = self.index.search('red')['items'][0]['id']
        self.index.config['openclip_pretrained'] = 'new-weights'
        original = self.encoder.image
        def remove(path, config):
            if path == self.red:
                self.index.remove_image(image_id)
            return original(path, config)
        with patch.object(self.encoder, 'image', side_effect=remove):
            self.index.scan()
        self.assertEqual(self.index.search()['total'], 1)
        self.index.config['openclip_pretrained'] = 'another-weights'
        def clear(path, config):
            self.index.clear()
            return original(path, config)
        with patch.object(self.encoder, 'image', side_effect=clear):
            self.index.scan()
        self.assertEqual(self.index.search()['total'], 0)

    def test_model_failure_stops_hybrid_before_llama(self):
        self.index.config['analysis_backend'] = 'hybrid'
        with patch.object(self.encoder, 'image', side_effect=OpenClipError('No model')), \
             patch.object(self.index, '_process_llama') as llama:
            with self.assertRaises(OpenClipError):
                self.index.scan()
            llama.assert_not_called()
        self.assertEqual(self.index.state['progress']['phase'], 'error')

    def test_http_semantic_search_and_embedding_status(self):
        from fastapi.testclient import TestClient
        from imageanalyser.app import create_app
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        app.state.index.encoder = self.encoder
        app.state.index.scan()
        client = TestClient(app)
        result = client.get('/api/search', params={'q': 'red', 'mode': 'semantic'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['items'][0]['path'], str(self.red))
        self.assertEqual(client.get('/api/status').json()['embedding_count'], 2)
        self.assertEqual(client.get('/api/search?mode=invalid').status_code, 422)
        self.assertEqual(client.get('/api/settings').json()['analysis_backend'], 'openclip')

    def test_manual_details_survive_embedding_rebuild_and_move(self):
        self.index.scan()
        item = self.index.search('red')['items'][0]
        corrected = {**item['analysis'], 'description': 'Corrected jacket', 'objects': ['velvet jacket']}
        self.index.edit_image(item['id'], corrected, 'Manual prompt\nSteps: 30, Seed: 555')
        self.index.config['openclip_pretrained'] = 'different-weights'
        self.index.scan()
        changed = self.index.search('555')['items'][0]
        self.assertTrue(changed['manually_edited'])
        self.assertEqual(changed['analysis']['description'], 'Corrected jacket')
        self.assertTrue(changed['embedded'])
        moved = self.root / 'moved.png'
        self.red.rename(moved)
        self.index.scan()
        moved_item = self.index.search('moved')['items'][0]
        self.assertTrue(moved_item['manually_edited'])
        self.assertIn('555', moved_item['a1111_metadata']['raw'])

    def test_tooltip_endpoint_returns_full_vector_without_model_inference(self):
        from fastapi.testclient import TestClient
        from imageanalyser.app import create_app
        self.index.scan()
        image_id = self.index.search('red')['items'][0]['id']
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        with patch.object(app.state.index.encoder, 'load') as load:
            response = client.get(f'/api/images/{image_id}/openclip')
            load.assert_not_called()
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['embedding'], [1.0, 0.0])
        self.assertEqual(data['dimensions'], 2)
        self.assertEqual(data['model'], 'ViT-B-32')
        self.assertEqual(data['pretrained_weights'], 'laion2b_s34b_b79k')
        self.assertTrue(data['matches_current_model'])
        self.assertTrue(data['matches_file_hash'])
        self.assertEqual(data['sha256'], self.index.hash_file(self.red))
        app.state.index.config['openclip_pretrained'] = 'other-weights'
        self.assertFalse(client.get(f'/api/images/{image_id}/openclip').json()['matches_current_model'])
        self.assertEqual(client.get('/api/images/99999/openclip').status_code, 404)
        self.index.remove_image(image_id)
        self.assertEqual(client.get(f'/api/images/{image_id}/openclip').status_code, 404)

    def test_tooltip_rejects_invalid_vectors(self):
        from fastapi.testclient import TestClient
        from imageanalyser.app import create_app
        self.index.scan()
        image_id = self.index.search('red')['items'][0]['id']
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        for raw in (b'x', vector(float('nan'), 0)):
            with self.index.connect() as db:
                db.execute('UPDATE images SET image_embedding=? WHERE id=?', (raw, image_id))
            self.assertEqual(client.get(f'/api/images/{image_id}/openclip').status_code, 409)
        with self.index.connect() as db:
            db.execute('UPDATE images SET image_embedding=NULL WHERE id=?', (image_id,))
        self.assertEqual(client.get(f'/api/images/{image_id}/openclip').status_code, 404)

    def test_backup_restores_embeddings_and_rejects_incompatible_search_mode(self):
        self.index.scan()
        backup = self.index.backup_database()
        self.addCleanup(backup.unlink, missing_ok=True)
        self.index.clear()
        self.index.restore_database(backup)
        self.assertEqual(self.index.search('red', mode='semantic')['total'], 2)
        self.assertTrue(self.index.paused)
        self.index.config['analysis_backend'] = 'llama'
        with self.assertRaises(OpenClipError):
            self.index.search('red', mode='semantic')
        self.assertEqual(self.index.search('studio', mode='hybrid')['total'], 1)
