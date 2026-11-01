from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from imageanalyser.app import create_app
from imageanalyser.diagnostics import read_events


class LogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'imageanalyser.log'

    def test_error_filter_preserves_complete_traceback_and_search_context(self):
        self.path.write_text('2026-11-01 08:00:00,000 INFO Scan started\n'
                             '2026-11-01 08:00:01,000 ERROR Analysis failed for photo.png\n'
                             'Traceback (most recent call last):\n'
                             '  File "core.py", line 1\nValueError: Invalid result\n'
                             '2026-11-01 08:00:02,000 INFO OpenCLIP indexed: other.png\n')
        result = read_events(self.path, category='error', query='ValueError')
        self.assertEqual(result['matched'], 1)
        self.assertEqual(len(result['entries']), 1)
        self.assertIn('photo.png', result['entries'][0]['text'])
        self.assertIn('Traceback', result['entries'][0]['text'])
        self.assertIn('ValueError: Invalid result', result['entries'][0]['text'])
        self.assertNotIn('other.png', result['entries'][0]['text'])

    def test_filters_search_rotations_before_limiting_results(self):
        Path(str(self.path)+'.1').write_text('2026-11-01 08:00:00,000 WARNING Root unavailable: network\n'
                                           '2026-11-01 08:00:01,000 ERROR Older failure\n')
        self.path.write_text('2026-11-01 08:01:00,000 INFO OpenCLIP indexed: a.png\n'
                             '2026-11-01 08:01:01,000 INFO [success] Indexed image in 1s: b.png\n'
                             '2026-11-01 08:01:02,000 INFO Scan finished\n')
        problems = read_events(self.path, limit=1, category='problems')
        self.assertEqual(problems['matched'], 2)
        self.assertEqual(len(problems['entries']), 1)
        self.assertIn('Older failure', problems['entries'][0]['text'])
        self.assertEqual(read_events(self.path, category='success')['matched'], 2)
        self.assertEqual(read_events(self.path, category='activity')['matched'], 1)
        self.assertEqual(read_events(self.path, category='warning', query='NETWORK')['matched'], 1)
        self.assertEqual(read_events(self.path, query='not present')['entries'], [])

    def test_api_validation_and_new_records_have_categories(self):
        app = create_app({'roots': [str(self.root)], 'data_dir': str(self.root / 'data'),
                          'llama_url': 'http://localhost:8080/v1', 'model': 'test'})
        index = app.state.index
        self.addCleanup(index.log_handler.close)
        index.log.info('OpenCLIP indexed: photo.png')
        index.log.info('Scan started')
        index.log.warning('Mount unavailable')
        client = TestClient(app)
        response = client.get('/api/logs', params={'category': 'success', 'q': 'PHOTO'})
        self.assertEqual(response.status_code, 200)
        entry = response.json()['entries'][0]
        self.assertEqual(entry['category'], 'success')
        self.assertIn('[success]', entry['text'])
        self.assertEqual(client.get('/api/logs?category=invalid').status_code, 422)
        self.assertEqual(client.get('/api/logs?limit=1001').status_code, 422)
        self.assertEqual(client.get('/api/logs', params={'q': 'x'*501}).status_code, 422)
        self.assertEqual(client.get('/api/logs?category=problems').json()['matched'], 1)

    def test_missing_files_and_incomplete_leading_tracebacks(self):
        self.assertEqual(read_events(self.path)['entries'], [])
        self.path.write_text('orphaned traceback line\n'
                             '2026-11-01 08:01:00,000 INFO [activity] Scanner resumed\n')
        result = read_events(self.path)
        self.assertEqual(result['matched'], 1)
        self.assertNotIn('orphaned', result['entries'][0]['text'])
