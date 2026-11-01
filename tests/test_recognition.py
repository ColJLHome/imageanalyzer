from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from imageanalyser.core import Index
from imageanalyser.recognition import RecognitionError, recognition_key


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'photos'
        self.root.mkdir()
        self.red, self.blue = self.root / 'red.png', self.root / 'blue.png'
        Image.new('RGB', (30, 30), 'red').save(self.red)
        Image.new('RGB', (30, 30), 'blue').save(self.blue)
        self.config = {'roots': [str(self.root)], 'data_dir': str(self.base / 'data'),
                       'llama_url': 'http://localhost:8080/v1', 'model': 'test',
                       'analysis_backend': 'hybrid', 'recognize_enabled': True}
        self.index = Index(self.config)
        self.addCleanup(self.index.log_handler.close)
        self.events = []
        self.tags = {'tags': ['jacket', 'person'], 'tags_chinese': ['人']}
        self.caption = {'description': 'Original caption', 'scenes': ['studio'],
                        'objects': ['car'], 'colours': ['blue'], 'people': []}
        def embed(path, config):
            self.events.append(('embedding', path))
            return struct.pack('<ff', 1, 0)
        def recognize(path, config):
            self.events.append(('recognizing', path))
            return self.tags
        def analyse(path):
            self.events.append(('caption', path))
            _, palette, width, height = self.index.prepare(path)
            return self.caption, palette, width, height
        self.patches = [patch.object(self.index.encoder, 'image', side_effect=embed),
                        patch.object(self.index.recognizer, 'image', side_effect=recognize),
                        patch.object(self.index, 'analyse', side_effect=analyse)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_all_embeddings_then_all_ram_tags_then_all_captions(self):
        self.index.scan()
        self.assertEqual([event[0] for event in self.events],
                         ['embedding', 'embedding', 'recognizing', 'recognizing', 'caption', 'caption'])
        self.assertEqual(self.index.search('jacket')['total'], 2)
        self.assertEqual(self.index.search('人')['total'], 2)
        self.assertEqual(self.index.search()['items'][0]['analysis'], self.caption)
        self.assertEqual(self.index.search()['items'][0]['ram_data'], self.tags)
        self.events.clear()
        self.index.scan()
        self.assertEqual(self.events, [])

    def test_ram_only_pass_in_openclip_mode_never_calls_llama(self):
        self.index.config['analysis_backend'] = 'openclip'
        self.index.scan()
        self.assertEqual([event[0] for event in self.events], ['embedding', 'embedding', 'recognizing', 'recognizing'])
        self.assertEqual(self.index.search('jacket')['total'], 2)

    def test_ram_cache_reused_for_copies_and_survives_restart(self):
        self.index.scan()
        self.events.clear()
        copied = self.root / 'copy.png'
        shutil.copyfile(self.red, copied)
        self.index.scan()
        self.assertEqual(self.events, [])
        self.assertEqual(self.index.search('jacket')['total'], 3)
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        with patch.object(restarted.recognizer, 'image') as recognize:
            restarted.scan()
            recognize.assert_not_called()

    def test_checkpoint_change_retags_and_manual_corrections_are_preserved(self):
        self.index.scan()
        item = self.index.search('red')['items'][0]
        self.index.edit_image(item['id'], self.caption, '', ram_tags=['corrected coat'])
        self.events.clear()
        self.index.config['recognize_checkpoint'] = str(self.base / 'custom.pth')
        self.index.scan()
        calls = [path for event, path in self.events if event == 'recognizing']
        self.assertEqual(calls, [self.blue])
        self.assertEqual(self.index.search('corrected coat')['total'], 1)
        corrected = self.index.search('corrected coat')['items'][0]
        self.assertTrue(corrected['ram_manually_edited'])
        self.assertTrue(corrected['embedded'])

    def test_ram_failure_stops_pipeline_before_caption_pass(self):
        with patch.object(self.index.recognizer, 'image', side_effect=RecognitionError('Cannot load checkpoint')):
            with self.assertRaises(RecognitionError):
                self.index.scan()
        self.assertFalse(any(event == 'caption' for event, _ in self.events))
        self.assertEqual(self.index.state['progress']['phase'], 'error')
        with self.index.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM images WHERE image_embedding IS NOT NULL').fetchone()[0], 2)

    def test_remove_during_ram_inference_cannot_restore_record(self):
        self.index.config['analysis_backend'] = 'openclip'
        def remove(path, config):
            with self.index.connect() as db:
                image_id = db.execute('SELECT id FROM images WHERE path=?', (str(path),)).fetchone()['id']
            self.index.remove_image(image_id)
            return self.tags
        with patch.object(self.index.recognizer, 'image', side_effect=remove):
            self.index.scan()
        self.assertEqual(self.index.search()['total'], 0)

    def test_ram_tags_remain_searchable_when_llama_is_unavailable_without_openclip(self):
        self.index.config['analysis_backend'] = 'llama'
        with patch.object(self.index, 'analyse', side_effect=RuntimeError('Caption server offline')):
            self.index.scan()
        self.assertEqual(self.index.search('jacket')['total'], 2)
        self.assertFalse(any(event == 'embedding' for event, _ in self.events))

    def test_manual_ram_edit_during_inference_is_preserved(self):
        def correct(path, config):
            with self.index.connect() as db:
                image_id = db.execute('SELECT id FROM images WHERE path=?', (str(path),)).fetchone()['id']
            self.index.edit_image(image_id, self.caption, '', ram_tags=['curated tag'])
            return self.tags
        with patch.object(self.index.recognizer, 'image', side_effect=correct):
            self.index.scan()
        self.assertEqual(self.index.search('curated tag')['total'], 2)
        self.assertEqual(self.index.search('jacket')['total'], 0)

    def test_backup_restore_preserves_ram_and_edit_api(self):
        from fastapi.testclient import TestClient
        from imageanalyser.app import create_app
        self.index.scan()
        backup = self.index.backup_database()
        self.addCleanup(backup.unlink, missing_ok=True)
        self.index.clear()
        self.index.restore_database(backup)
        self.assertEqual(self.index.search('jacket')['total'], 2)
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        self.assertEqual(client.get('/api/status').json()['recognition_count'], 2)
        item = self.index.search()['items'][0]
        response = client.put(f"/api/images/{item['id']}", json={**self.caption, 'ram_tags': ['manual hat']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(client.get('/api/search?q=manual%20hat').json()['total'], 1)
        self.assertEqual(client.get('/api/settings').json()['recognize_enabled'], True)
        with self.index.connect() as db:
            row = db.execute('SELECT ram_model FROM images WHERE id=?', (item['id'],)).fetchone()
            self.assertEqual(row['ram_model'], recognition_key(self.config))
