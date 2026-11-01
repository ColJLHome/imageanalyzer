from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from imageanalyser.core import Index


class ScanBadgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'photos'
        self.root.mkdir()
        self.red, self.blue = self.root / 'red.png', self.root / 'blue.png'
        info = PngInfo()
        info.add_text('parameters', 'red jacket\nSteps: 20, Seed: 42')
        Image.new('RGB', (30, 30), 'red').save(self.red, pnginfo=info)
        Image.new('RGB', (30, 30), 'blue').save(self.blue)
        self.config = {'roots': [str(self.root)], 'data_dir': str(Path(self.temp.name) / 'data'),
                       'llama_url': 'http://localhost:8080/v1', 'model': 'caption-model',
                       'analysis_backend': 'hybrid', 'recognize_enabled': True}
        self.index = Index(self.config)
        self.addCleanup(self.index.log_handler.close)
        self.analysis = {'description': 'A photo', 'scenes': [], 'objects': [], 'colours': [], 'people': []}
        def caption(path):
            _, palette, w, h = self.index.prepare(path)
            return self.analysis, palette, w, h
        for item in (patch.object(self.index.encoder, 'image', return_value=struct.pack('<ff', 1, 0)),
                     patch.object(self.index.recognizer, 'image', return_value={'tags': ['jacket'], 'tags_chinese': []}),
                     patch.object(self.index, 'analyse', side_effect=caption)):
            item.start()
            self.addCleanup(item.stop)

    def test_all_completed_scans_reported_with_their_provenance(self):
        self.index.scan()
        item = self.index.search('red')['items'][0]
        self.assertEqual(set(item['scans']), {'openclip', 'a1111', 'ram', 'llama'})
        self.assertTrue(item['scans']['a1111']['metadata_saved'])
        self.assertTrue(item['scans']['openclip']['matches_current_model'])
        self.assertEqual(item['scans']['llama']['model'], 'caption-model')
        self.assertEqual(item['ram_data']['tags'], ['jacket'])
        self.assertIn('Seed: 42', item['a1111_metadata']['raw'])
        blue = self.index.search('blue')['items'][0]
        self.assertIn('a1111', blue['scans'])
        self.assertFalse(blue['scans']['a1111']['metadata_saved'])

    def test_placeholder_and_manual_data_do_not_claim_llama_analysis(self):
        self.index.config['analysis_backend'] = 'openclip'
        self.index.scan()
        item = self.index.search('red')['items'][0]
        self.assertNotIn('llama', item['scans'])
        self.index.edit_image(item['id'], self.analysis, item['a1111_metadata']['raw'])
        edited = self.index.search('red')['items'][0]
        self.assertNotIn('llama', edited['scans'])
        self.assertTrue(edited['manually_edited'])

    def test_old_stored_model_still_has_badge_with_model_mismatch_flag(self):
        self.index.scan()
        self.index.config['openclip_pretrained'] = 'new-model'
        self.index.config['recognize_checkpoint'] = str(self.root / 'different-checkpoint.pth')
        item = self.index.search('red')['items'][0]
        self.assertFalse(item['embedded'])
        self.assertIn('openclip', item['scans'])
        self.assertFalse(item['scans']['openclip']['matches_current_model'])
        self.assertFalse(item['scans']['ram']['matches_current_model'])

    def test_existing_metadata_checks_are_migrated(self):
        self.index.scan()
        with self.index.connect() as db:
            db.execute('ALTER TABLE images DROP COLUMN a1111_sha256')
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        for item in restarted.search()['items']:
            self.assertIn('a1111', item['scans'])

    def test_manual_ram_tags_without_inference_do_not_claim_ram_scan(self):
        self.index.config.update(analysis_backend='openclip', recognize_enabled=False)
        self.index.scan()
        item = self.index.search('red')['items'][0]
        self.index.edit_image(item['id'], self.analysis, '', ram_tags=['curated tag'])
        edited = self.index.search('red')['items'][0]
        self.assertNotIn('ram', edited['scans'])
        self.assertEqual(edited['ram_data']['tags'], ['curated tag'])

    def test_failed_caption_does_not_show_completed_llama_badge(self):
        with patch.object(self.index, 'analyse', side_effect=RuntimeError('Caption failed')):
            self.index.scan()
        for item in self.index.search()['items']:
            self.assertNotIn('llama', item['scans'])
            self.assertEqual(set(item['scans']), {'openclip', 'a1111', 'ram'})
