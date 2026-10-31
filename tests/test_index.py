import json
import hashlib
import os
import shutil
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from imageanalyser.app import create_app
from imageanalyser.core import Index, parse_analysis


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'photos'
        (self.root / 'nested').mkdir(parents=True)
        self.file = self.root / 'nested' / 'car.png'
        Image.new('RGB', (100, 50), 'red').save(self.file)
        self.config = {'roots': [str(self.root)], 'data_dir': str(self.base / 'data'),
                       'llama_url': 'http://localhost:8080/v1', 'model': 'test'}
        self.index = Index(self.config)
        self.addCleanup(self.index.log_handler.close)
        self.result = {'description': 'A red car on a beach', 'objects': ['red car'],
                       'scenes': ['beach'], 'colours': ['red']}

    def fake_analysis(self, path):
        _, palette, width, height = self.index.prepare(path)
        return self.result, palette, width, height

    def test_recursive_incremental_and_filters(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis) as call:
            self.index.scan()
            self.index.scan()
            self.assertEqual(call.call_count, 1)
            self.assertEqual(self.index.search('red car', scene='beach')['total'], 1)
            self.assertEqual(self.index.search(colour='blue')['total'], 0)
            self.assertEqual(self.index.search(object_name='red car')['total'], 1)
            self.assertEqual(self.index.search(q='%')['total'], 0)
            Image.new('RGB', (90, 60), 'blue').save(self.file)
            self.index.scan()
            self.assertEqual(call.call_count, 2)
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        with patch.object(restarted, 'analyse') as call:
            restarted.scan()
            call.assert_not_called()
        self.assertEqual(restarted.search()['total'], 1)

    def test_errors_retry_and_recover(self):
        with patch.object(self.index, 'analyse', side_effect=RuntimeError('offline')) as call:
            self.index.scan()
            self.index.scan()
            self.assertEqual(call.call_count, 1)
        self.assertEqual(self.index.search()['total'], 0)
        with self.index.connect() as db:
            db.execute('UPDATE images SET retry_at=0')
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        self.assertEqual(self.index.search()['total'], 1)

    def test_unavailable_mount_retains_index_and_symlinks_skipped(self):
        outside = self.base / 'outside.png'
        Image.new('RGB', (20, 20)).save(outside)
        (self.root / 'link.png').symlink_to(outside)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis) as call:
            self.index.scan()
            self.assertEqual(call.call_count, 1)
        self.root.rename(self.base / 'offline')
        self.index.scan()
        self.assertEqual(self.index.search()['total'], 1)
        self.assertIn('Unavailable', self.index.state['roots'][str(self.root)])
        with self.assertRaises(ValueError):
            self.index.permitted(outside)

    def test_api_and_thumbnail_boundary(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        result = client.get('/api/search', params={'object': 'red car'}).json()
        image_id = result['items'][0]['id']
        self.assertEqual(client.get(f'/api/images/{image_id}/thumbnail').status_code, 200)
        self.assertEqual(client.get('/api/search?offset=-1').status_code, 422)
        self.assertEqual(client.get('/api/images/9999/thumbnail').status_code, 404)
        self.file.unlink()
        self.file.symlink_to(self.base / 'secret.png')
        Image.new('RGB', (20, 20)).save(self.base / 'secret.png')
        self.assertEqual(client.get(f'/api/images/{image_id}/thumbnail').status_code, 404)
        self.assertEqual(client.get('/').status_code, 200)

    def test_llama_request_and_validation(self):
        response = type('Response', (), {'raise_for_status': lambda self: None,
            'json': lambda self_: {'choices': [{'message': {'content': json.dumps(self.result)}}]}})()
        with patch('imageanalyser.core.httpx.post', return_value=response) as post:
            result, palette, width, height = self.index.analyse(self.file)
        self.assertEqual(result, self.result)
        self.assertEqual((width, height), (100, 50))
        self.assertEqual(palette[0]['hex'], '#ff0000')
        self.assertEqual(post.call_args.args[0], 'http://localhost:8080/v1/chat/completions')
        request = post.call_args.kwargs['json']
        self.assertEqual(request['response_format']['type'], 'json_object')
        self.assertEqual(request['response_format']['schema']['required'],
                         ['description', 'scenes', 'objects', 'colours'])
        self.assertFalse(request['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(request['max_tokens'], 2048)
        content = request['messages'][0]['content']
        self.assertTrue(content[1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        with self.assertRaises(ValueError):
            parse_analysis('{"description":"bad","objects":"car"}')

    def test_model_response_errors_are_actionable_and_logged(self):
        cases = [
            ({'content': '', 'reasoning_content': 'Still thinking'}, 'length', 'output token limit'),
            ({'content': ''}, 'stop', 'no final answer'),
            ({'content': 'Here is my description'}, 'stop', 'Invalid model answer'),
            ({'content': None}, 'stop', 'no final answer'),
        ]
        for message, finish, expected in cases:
            with self.subTest(message=message, finish=finish):
                data = {'choices': [{'message': message, 'finish_reason': finish}],
                        'usage': {'completion_tokens': 2048}}
                response = type('Response', (), {'raise_for_status': lambda self: None,
                    'json': lambda self_: data})()
                with patch('imageanalyser.core.httpx.post', return_value=response):
                    with self.assertRaisesRegex(ValueError, expected):
                        self.index.analyse(self.file)
        logs = '\n'.join(self.index.logs())
        self.assertIn('finish_reason=length', logs)
        self.assertIn('reasoning_chars=14', logs)
        self.assertIn('Here is my description', logs)
        self.assertNotIn('data:image', logs)
        self.assertNotIn('Still thinking', logs)

    def test_parse_wrapped_json_and_reject_incomplete_answers(self):
        valid = json.dumps(self.result)
        self.assertEqual(parse_analysis('```json\n' + valid + '\n```'), self.result)
        self.assertEqual(parse_analysis('<think>Look carefully</think>\n' + valid), self.result)
        for invalid in ('', '<think>unfinished', '```json', '{"description":"truncated'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_analysis(invalid)

    def test_settings_save_validate_and_persist_without_overwriting_service_config(self):
        app = create_app(self.config)
        index = app.state.index
        self.addCleanup(index.log_handler.close)
        client = TestClient(app)
        settings = client.get('/api/settings').json()
        new_root = self.base / 'new photos'
        new_root.mkdir()
        values = {**settings, 'roots': [str(new_root)], 'model': 'new-vision',
                  'llama_url': 'http://ai-server:8080/v1/', 'max_output_tokens': 4096}
        response = client.put('/api/settings', json=values)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(index.config['model'], 'new-vision')
        self.assertEqual(index.config['llama_url'], 'http://ai-server:8080/v1')
        self.assertTrue(index.wake.is_set())
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        self.assertEqual(restarted.get_settings(), response.json())
        self.assertEqual(restarted.roots, [new_root])
        self.assertEqual(restarted.data, self.index.data)
        self.assertEqual(self.config['model'], 'test')
        for invalid in ({'roots': []}, {'roots': ['relative/path']}, {'model': '  '},
                        {'llama_url': 'file:///tmp/server'}, {'llama_url': 'http://user:pass@server/v1'},
                        {'request_timeout_seconds': 0}, {'max_output_tokens': 0}):
            with self.subTest(invalid=invalid):
                self.assertEqual(client.put('/api/settings', json={**values, **invalid}).status_code, 422)
                self.assertEqual(index.get_settings(), response.json())
        with self.assertRaises(ValueError):
            index.permitted(self.file)

    def test_settings_change_discards_old_result_and_preserves_pause(self):
        from imageanalyser.settings import RuntimeSettings
        def change_settings(path):
            self.index.save_settings(RuntimeSettings(**{**self.index.get_settings(), 'model': 'replacement'}))
            return self.fake_analysis(path)
        with patch.object(self.index, 'analyse', side_effect=change_settings):
            self.index.scan()
        self.assertEqual(self.index.search()['total'], 0)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        with self.index.connect() as db:
            self.assertEqual(db.execute('SELECT model FROM images').fetchone()['model'], 'replacement')
        self.index.clear()
        self.index.save_settings(RuntimeSettings(**self.index.get_settings()))
        self.assertTrue(self.index.paused)

    def test_hash_reuses_analysis_after_move_and_restart(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        expected_hash = hashlib.sha256(self.file.read_bytes()).hexdigest()
        with self.index.connect() as db:
            self.assertEqual(db.execute('SELECT sha256 FROM images').fetchone()['sha256'], expected_hash)
        moved = self.root / 'renamed-view.png'
        self.file.rename(moved)
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        with patch.object(restarted, 'analyse') as analyse:
            restarted.scan()
            analyse.assert_not_called()
        result = restarted.search('renamed-view')['items'][0]
        self.assertEqual(result['path'], str(moved))
        self.assertEqual(result['analysis'], self.result)
        self.assertIn('Reused analysis by SHA-256', '\n'.join(restarted.logs()))

    def test_hash_reuses_copies_and_timestamp_changes_but_not_changed_content(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis) as analyse:
            self.index.scan()
            copied = self.root / 'copy.png'
            shutil.copyfile(self.file, copied)
            self.index.scan()
            self.assertEqual(analyse.call_count, 1)
            before = copied.stat()
            os.utime(copied, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
            self.index.scan()
            self.assertEqual(analyse.call_count, 1)
            Image.new('RGB', (100, 50), 'green').save(copied)
            self.index.scan()
            self.assertEqual(analyse.call_count, 2)
        with self.index.connect() as db:
            hashes = {r['sha256'] for r in db.execute("SELECT sha256 FROM images WHERE status='done'")}
            self.assertEqual(len(hashes), 2)

    def test_legacy_database_migrates_and_backfills_without_analysis(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        with self.index.connect() as db:
            db.execute('DROP INDEX images_sha256')
            db.execute('ALTER TABLE images DROP COLUMN sha256')
        migrated = Index(self.config)
        self.addCleanup(migrated.log_handler.close)
        with patch.object(migrated, 'analyse') as analyse:
            migrated.scan()
            analyse.assert_not_called()
        with migrated.connect() as db:
            self.assertEqual(db.execute('SELECT sha256 FROM images').fetchone()['sha256'],
                             hashlib.sha256(self.file.read_bytes()).hexdigest())
            indexes = {r['name'] for r in db.execute('PRAGMA index_list(images)')}
            self.assertIn('images_sha256', indexes)
        self.assertEqual(migrated.search()['total'], 1)

    def test_removed_analysis_is_not_reused_for_another_path(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        self.index.remove_image(self.index.search()['items'][0]['id'])
        copied = self.root / 'copy.png'
        shutil.copyfile(self.file, copied)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis) as analyse:
            self.index.scan()
            self.assertEqual(analyse.call_count, 1)
        self.assertEqual(self.index.search()['total'], 1)
        self.assertEqual(self.index.search()['items'][0]['path'], str(copied))

    def test_scan_progress_counts_checked_and_remaining_including_skips(self):
        second = self.root / 'second.png'
        Image.new('RGB', (30, 30), 'blue').save(second)
        snapshots = []
        def analyse(path):
            snapshots.append(dict(self.index.state['progress']))
            return self.fake_analysis(path)
        with patch.object(self.index, 'analyse', side_effect=analyse):
            self.index.scan()
        self.assertEqual([p['total'] for p in snapshots], [2, 2])
        self.assertEqual([p['scanned'] for p in snapshots], [0, 1])
        progress = self.index.state['progress']
        self.assertEqual(progress, {'phase': 'complete', 'scanned': 2, 'total': 2, 'discovered': 2})
        with patch.object(self.index, 'analyse') as analyse:
            self.index.scan()
            analyse.assert_not_called()
        self.assertEqual(self.index.state['progress']['scanned'], 2)
        self.assertEqual(self.index.state['progress']['total'], 2)

    def test_scan_progress_pauses_when_index_cleared_during_analysis(self):
        def clear(path):
            self.index.clear()
            return self.fake_analysis(path)
        with patch.object(self.index, 'analyse', side_effect=clear):
            self.index.scan()
        self.assertEqual(self.index.state['progress']['phase'], 'paused')
        self.assertEqual(self.index.state['progress']['scanned'], 0)
        self.assertEqual(self.index.state['progress']['total'], 1)
        self.assertFalse(self.index.state['running'])

    def test_database_backup_restore_roundtrip_and_safety_backup(self):
        app = create_app(self.config)
        index = app.state.index
        self.addCleanup(index.log_handler.close)
        client = TestClient(app)
        with patch.object(index, 'analyse', side_effect=self.fake_analysis):
            index.scan()
        from imageanalyser.settings import RuntimeSettings
        index.save_settings(RuntimeSettings(**{**index.get_settings(), 'model': 'saved-model'}))
        downloaded = client.get('/api/database/backup')
        self.assertEqual(downloaded.status_code, 200)
        self.assertTrue(downloaded.content.startswith(b'SQLite format 3'))
        image_id = index.search()['items'][0]['id']
        index.remove_image(image_id)
        response = client.post('/api/database/restore?confirm=true', content=downloaded.content,
                               headers={'Content-Type': 'application/octet-stream'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(index.search()['total'], 1)
        self.assertEqual(index.get_settings()['model'], 'saved-model')
        self.assertTrue(index.paused)
        safety = index.data / 'backups' / response.json()['safety_backup']
        self.assertTrue(safety.exists())
        import sqlite3
        db = sqlite3.connect(safety)
        try:
            self.assertEqual(db.execute('SELECT status FROM images').fetchone()[0], 'removed')
        finally:
            db.close()
        self.assertTrue(self.file.exists())

    def test_invalid_restore_does_not_change_current_database(self):
        app = create_app(self.config)
        index = app.state.index
        self.addCleanup(index.log_handler.close)
        client = TestClient(app)
        with patch.object(index, 'analyse', side_effect=self.fake_analysis):
            index.scan()
        self.assertEqual(client.post('/api/database/restore', content=b'invalid').status_code, 400)
        self.assertEqual(client.post('/api/database/restore?confirm=true', content=b'invalid').status_code, 400)
        self.assertEqual(client.post('/api/database/restore?confirm=true', content=b'').status_code, 400)
        self.assertEqual(index.search()['total'], 1)
        self.assertFalse(index.paused)
        self.assertFalse((index.data / 'backups').exists())

    def test_restore_during_analysis_discards_inflight_result(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        backup = self.index.backup_database()
        self.addCleanup(backup.unlink, missing_ok=True)
        Image.new('RGB', (50, 50), 'blue').save(self.file)
        def restore_during_analysis(path):
            self.index.restore_database(backup)
            return {**self.result, 'description': 'Should be discarded'}, [], 50, 50
        with patch.object(self.index, 'analyse', side_effect=restore_during_analysis):
            self.index.scan()
        self.assertEqual(self.index.search()['items'][0]['analysis'], self.result)
        self.assertTrue(self.index.paused)

    def test_individual_removal_persists_and_full_clear_resets_exclusions(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        image_id = client.get('/api/search').json()['items'][0]['id']
        self.assertEqual(client.request('DELETE', f'/api/images/{image_id}', json={'confirm': False}).status_code, 422)
        self.assertEqual(client.request('DELETE', f'/api/images/{image_id}', json={'confirm': True}).status_code, 200)
        self.assertEqual(client.get('/api/search').json()['total'], 0)
        self.assertEqual(client.get('/api/status').json()['counts'], {})
        self.assertEqual(client.get(f'/api/images/{image_id}/thumbnail').status_code, 404)
        self.assertEqual(client.request('DELETE', '/api/images/9999', json={'confirm': True}).status_code, 404)
        self.assertTrue(self.file.exists())
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        # Even a changed file stays excluded.
        Image.new('RGB', (40, 40), 'blue').save(self.file)
        with patch.object(restarted, 'analyse') as analyse:
            restarted.scan()
            analyse.assert_not_called()
        with restarted.connect() as db:
            record = db.execute('SELECT * FROM images WHERE id=?', (image_id,)).fetchone()
            self.assertEqual(record['status'], 'removed')
            self.assertIsNone(record['analysis'])
        restarted.clear()
        restarted.resume()
        with patch.object(restarted, 'analyse', side_effect=self.fake_analysis):
            restarted.scan()
        self.assertEqual(restarted.search()['total'], 1)

    def test_removal_during_analysis_cannot_restore_result_or_error(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.index.clear()
                self.index.resume()
                def remove_during_analysis(path):
                    with self.index.connect() as db:
                        image_id = db.execute('SELECT id FROM images WHERE path=?', (str(path),)).fetchone()['id']
                    self.assertTrue(self.index.remove_image(image_id))
                    if fail:
                        raise RuntimeError('Analysis failed after removal')
                    return self.fake_analysis(path)
                with patch.object(self.index, 'analyse', side_effect=remove_during_analysis):
                    self.index.scan()
                self.assertEqual(self.index.search()['total'], 0)
                with self.index.connect() as db:
                    record = db.execute('SELECT status,analysis,error FROM images').fetchone()
                    self.assertEqual(record['status'], 'removed')
                    self.assertIsNone(record['analysis'])
                    self.assertIsNone(record['error'])

    def test_clear_api_preserves_files_and_pause_survives_restart(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        self.assertEqual(client.post('/api/images/clear', json={'confirm': False}).status_code, 422)
        self.assertEqual(client.post('/api/images/clear', json={'confirm': True}).json()['cleared'], 1)
        self.assertTrue(self.file.exists())
        self.assertEqual(client.get('/api/search').json()['total'], 0)
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        self.assertTrue(restarted.paused)
        with patch.object(restarted, 'analyse') as analyse:
            restarted.scan()
            analyse.assert_not_called()
        self.assertTrue(client.get('/api/status').json()['paused'])
        self.assertEqual(client.post('/api/scan/resume', json={'confirm': True}).status_code, 200)
        with patch.object(app.state.index, 'analyse', side_effect=self.fake_analysis):
            app.state.index.scan()
        self.assertEqual(client.get('/api/search').json()['total'], 1)

    def test_clear_scanner_details_keeps_index_retries_and_logs(self):
        app = create_app(self.config)
        index = app.state.index
        self.addCleanup(index.log_handler.close)
        client = TestClient(app)
        with patch.object(index, 'analyse', side_effect=RuntimeError('Old failure')):
            index.scan()
        before = client.get('/api/status').json()
        self.assertEqual(len(before['errors']), 1)
        self.assertTrue(before['roots'])
        self.assertEqual(client.post('/api/scanner/details/clear', json={'confirm': False}).status_code, 422)
        self.assertEqual(client.post('/api/scanner/details/clear', json={'confirm': True}).status_code, 200)
        cleared = client.get('/api/status').json()
        self.assertEqual(cleared['errors'], [])
        self.assertEqual(cleared['roots'], {})
        self.assertIsNone(cleared['last_scan'])
        self.assertEqual(cleared['counts'], before['counts'])
        self.assertFalse(cleared['paused'])
        self.assertTrue(self.file.exists())
        self.assertIn('Old failure', '\n'.join(index.logs()))
        with index.connect() as db:
            row = db.execute('SELECT attempts,retry_at FROM images').fetchone()
            self.assertEqual(row['attempts'], 1)
            self.assertGreater(row['retry_at'], 0)
            db.execute('UPDATE images SET retry_at=0')
        with patch.object(index, 'analyse', side_effect=RuntimeError('New failure')):
            index.scan()
        fresh = client.get('/api/status').json()
        self.assertEqual(fresh['errors'][0]['error'], 'New failure')
        self.assertTrue(fresh['roots'])

    def test_scan_button_wakes_worker_and_only_analyses_new_images(self):
        app = create_app(self.config)
        index = app.state.index
        self.addCleanup(index.log_handler.close)
        client = TestClient(app)
        first_scan, second_scan = threading.Event(), threading.Event()
        scan = index.scan
        scans = []
        def tracked_scan():
            scan()
            scans.append(True)
            (first_scan if len(scans) == 1 else second_scan).set()
        with patch.object(index, 'analyse', side_effect=self.fake_analysis) as analyse, \
             patch.object(index, 'scan', side_effect=tracked_scan):
            worker = threading.Thread(target=index.run)
            worker.start()
            try:
                self.assertTrue(first_scan.wait(5))
                new_image = self.root / 'new.png'
                Image.new('RGB', (30, 30), 'blue').save(new_image)
                response = client.post('/api/scan', json={'confirm': True})
                self.assertEqual(response.status_code, 202)
                self.assertEqual(response.json()['status'], 'queued')
                self.assertTrue(second_scan.wait(5))
                self.assertEqual(analyse.call_count, 2)
                self.assertEqual(client.get('/api/search').json()['total'], 2)
            finally:
                index.stop.set()
                index.wake.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())

    def test_scan_request_resumes_pause_and_avoids_duplicate_active_scan(self):
        self.index.clear()
        self.assertEqual(self.index.request_scan(), 'queued')
        self.assertFalse(self.index.paused)
        self.assertTrue(self.index.wake.is_set())
        self.index.wake.clear()
        with self.index.scan_lock:
            self.assertEqual(self.index.request_scan(), 'running')
            self.assertFalse(self.index.wake.is_set())

    def test_clear_during_analysis_discards_inflight_result(self):
        started, finish = threading.Event(), threading.Event()
        def analyse(path):
            started.set()
            if not finish.wait(5):
                raise RuntimeError('Test analysis timed out')
            return self.fake_analysis(path)
        with patch.object(self.index, 'analyse', side_effect=analyse):
            worker = threading.Thread(target=self.index.scan)
            worker.start()
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(self.index.clear(), 1)
            finally:
                finish.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
        with self.index.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM images').fetchone()[0], 0)
        self.assertTrue(self.file.exists())

    def test_diagnostic_logs_include_errors_and_api_is_bounded(self):
        with patch.object(self.index, 'analyse', side_effect=RuntimeError('Model unavailable')):
            self.index.scan()
        logs = '\n'.join(self.index.logs())
        self.assertIn('Scan started', logs)
        self.assertIn('Analysing image:', logs)
        self.assertIn('Model unavailable', logs)
        self.assertIn('Traceback', logs)
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        lines = client.get('/api/logs?limit=2').json()['lines']
        self.assertEqual(len(lines), 2)
        self.assertEqual(client.get('/api/logs?limit=1001').status_code, 422)


if __name__ == '__main__':
    unittest.main()
