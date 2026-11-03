import json
import io
import struct
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

    def test_large_preview_dimensions_and_access_boundary(self):
        Image.new('RGB', (2400, 1200), 'red').save(self.file)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        image_id = client.get('/api/search').json()['items'][0]['id']
        for preview, expected in [(False, (480, 240)), (True, (1600, 800))]:
            response = client.get(f'/api/images/{image_id}/thumbnail', params={'preview': preview})
            self.assertEqual(response.status_code, 200)
            with Image.open(io.BytesIO(response.content)) as image:
                self.assertEqual(image.size, expected)
        self.file.unlink()
        Image.new('RGB', (20, 20)).save(self.base / 'secret.png')
        self.file.symlink_to(self.base / 'secret.png')
        self.assertEqual(client.get(f'/api/images/{image_id}/thumbnail?preview=true').status_code, 404)
        self.assertEqual(client.get('/api/images/9999/thumbnail?preview=true').status_code, 404)

    def test_llama_request_and_validation(self):
        response = type('Response', (), {'raise_for_status': lambda self: None,
            'json': lambda self_: {'choices': [{'message': {'content': json.dumps({**self.result, 'people': []})}}]}})()
        with patch('imageanalyser.core.httpx.post', return_value=response) as post:
            result, palette, width, height = self.index.analyse(self.file)
        self.assertEqual(result, {**self.result, 'people': []})
        self.assertEqual((width, height), (100, 50))
        self.assertEqual(palette[0]['hex'], '#ff0000')
        self.assertEqual(post.call_args.args[0], 'http://localhost:8080/v1/chat/completions')
        request = post.call_args.kwargs['json']
        self.assertEqual(request['response_format']['type'], 'json_object')
        self.assertEqual(request['response_format']['schema']['required'],
                         ['description', 'scenes', 'objects', 'colours', 'people'])
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

    def test_scan_orders_all_folders_by_name_size_and_creation_in_both_directions(self):
        self.file.unlink()
        paths = [self.root / 'zeta.png', self.root / 'nested' / 'Alpha.png', self.root / 'middle.png']
        for path, size in zip(paths, (5, 10, 15)):
            path.write_bytes(b'x' * size)
        births = dict(zip(paths, (300, 100, 200)))
        expected = {'name': [paths[1], paths[2], paths[0]],
                    'size': paths, 'created': [paths[1], paths[2], paths[0]]}
        for by in ('name', 'size', 'created'):
            for direction in ('asc', 'desc'):
                with self.subTest(by=by, direction=direction):
                    self.index.config.update(scan_sort_by=by, scan_sort_direction=direction)
                    with patch.object(self.index, 'process') as process, \
                         patch('imageanalyser.core.creation_time_ns', side_effect=lambda path, metadata: (births[path], False)):
                        self.index.scan()
                    actual = [call.args[0] for call in process.call_args_list]
                    self.assertEqual(actual, expected[by] if direction == 'asc' else list(reversed(expected[by])))

    def test_scan_order_settings_validate_and_persist(self):
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        settings = client.get('/api/settings').json()
        self.assertEqual(settings['scan_sort_by'], 'name')
        for invalid in ({'scan_sort_by': 'other'}, {'scan_sort_direction': 'other'}):
            self.assertEqual(client.put('/api/settings', json={**settings, **invalid}).status_code, 422)
        saved = {**settings, 'scan_sort_by': 'created', 'scan_sort_direction': 'desc'}
        self.assertEqual(client.put('/api/settings', json=saved).status_code, 200)
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        self.assertEqual(restarted.get_settings(), saved)

    def test_creation_time_uses_birth_time_and_explicit_mtime_fallback(self):
        from types import SimpleNamespace
        from imageanalyser.file_order import creation_time_ns
        import struct
        metadata = SimpleNamespace(st_mtime_ns=999, st_ctime_ns=12345)
        with patch('imageanalyser.file_order._statx', None):
            self.assertEqual(creation_time_ns(self.file, metadata), (999, True))
            self.assertEqual(creation_time_ns(self.file, SimpleNamespace(st_birthtime_ns=123, st_mtime_ns=999)), (123, False))
        def fake_statx(directory, path, flags, mask, buffer):
            struct.pack_into('=I', buffer, 0, 0x800)
            struct.pack_into('=qI', buffer, 0x50, 123, 456)
            return 0
        with patch('imageanalyser.file_order._statx', side_effect=fake_statx):
            self.assertEqual(creation_time_ns(self.file, metadata), (123_000_000_456, False))
        with patch('imageanalyser.file_order._statx', return_value=-1):
            self.assertEqual(creation_time_ns(self.file, metadata), (999, True))

    def test_sort_keeps_missing_files_at_end(self):
        missing = self.root / 'missing.png'
        for direction in ('asc', 'desc'):
            ordered = self.index.order_candidates([missing, self.file], 'size', direction, lambda: False)
            self.assertEqual(ordered, [self.file, missing])

    def test_people_details_are_stored_searchable_and_reused_by_hash(self):
        person = {'position': 'left', 'age_group': 'young adult', 'gender_presentation': 'feminine',
                  'clothing': ['blue striped jacket', 'black trousers', 'white trainers'],
                  'accessories': ['round glasses', 'brown shoulder bag']}
        analysis = {**self.result, 'people': [person]}
        self.assertEqual(parse_analysis(json.dumps(analysis)), analysis)
        def analyse(path):
            _, palette, width, height = self.fake_analysis(path)
            return analysis, palette, width, height
        with patch.object(self.index, 'analyse', side_effect=analyse) as call:
            self.index.scan()
            shutil.copyfile(self.file, self.root / 'copy.png')
            self.index.scan()
            self.assertEqual(call.call_count, 1)
        for query in ('striped jacket', 'young adult', 'feminine', 'round glasses', 'white trainers'):
            self.assertEqual(self.index.search(query)['total'], 2)
        self.assertEqual(self.index.search()['items'][0]['analysis']['people'], [person])
        for invalid in ({**analysis, 'people': 'person'},
                        {**analysis, 'people': [{**person, 'clothing': 'jacket'}]},
                        {**analysis, 'people': [{**person, 'age_group': '27'}]},
                        {**analysis, 'people': [{**person, 'gender_presentation': 'gender identity guess'}]}):
            with self.assertRaises(ValueError):
                parse_analysis(json.dumps(invalid))
        self.assertEqual(parse_analysis(json.dumps(self.result)), self.result)

    def test_restart_api_requires_supported_launcher_and_confirmation(self):
        unsupported = create_app(self.config)
        self.addCleanup(unsupported.state.index.log_handler.close)
        client = TestClient(unsupported)
        self.assertFalse(client.get('/api/status').json()['restart_supported'])
        self.assertEqual(client.post('/api/backend/restart', json={'confirm': True}).status_code, 503)
        calls = []
        app = create_app(self.config, restart_callback=lambda: calls.append(True))
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        self.assertEqual(client.post('/api/backend/restart', json={'confirm': False}).status_code, 422)
        self.assertFalse(app.state.index.stop.is_set())
        response = client.post('/api/backend/restart', json={'confirm': True})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()['instance_id'], app.state.instance_id)
        self.assertTrue(app.state.index.stop.is_set())
        self.assertEqual(calls, [True])
        self.assertEqual(client.post('/api/backend/restart', json={'confirm': True}).status_code, 202)
        self.assertEqual(calls, [True])

    def test_cli_backend_restarts_and_preserves_database_and_settings(self):
        import socket
        import subprocess
        import sys
        import time
        import httpx
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        with self.index.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES ('paused', 'true')")
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        config_path = self.base / 'config.json'
        config_path.write_text(json.dumps({**self.config, 'host': '127.0.0.1', 'port': port}))
        log_path = self.base / 'server-output.log'
        with log_path.open('w') as output:
            process = subprocess.Popen([sys.executable, '-m', 'imageanalyser.app', '--config', str(config_path)],
                                       stdout=output, stderr=subprocess.STDOUT)
            try:
                with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=1) as client:
                    def wait_for_backend(previous=None):
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline:
                            if process.poll() is not None:
                                self.fail('Backend exited: ' + log_path.read_text())
                            try:
                                response = client.get('/api/status')
                                response.raise_for_status()
                                data = response.json()
                                if previous is None or data['instance_id'] != previous:
                                    return data
                            except httpx.HTTPError:
                                pass
                            time.sleep(0.05)
                        self.fail('Backend did not reconnect: ' + log_path.read_text())
                    initial = wait_for_backend()
                    self.assertTrue(initial['paused'])
                    settings = client.get('/api/settings').json()
                    response = client.put('/api/settings', json={**settings, 'model': 'persistent-model'})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(client.post('/api/backend/restart', json={'confirm': True}).status_code, 202)
                    restarted = wait_for_backend(initial['instance_id'])
                    self.assertTrue(restarted['paused'])
                    self.assertEqual(client.get('/api/search').json()['total'], 1)
                    self.assertEqual(client.get('/api/settings').json()['model'], 'persistent-model')
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def test_a1111_png_metadata_is_stored_searchable_and_reused(self):
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        text = 'a sapphire velvet jacket, <lora:wardrobe:0.7>\nNegative prompt: blurry, watermark\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 7, Seed: 123456, Model: dream-model'
        info.add_text('parameters', text)
        Image.new('RGB', (100, 50), 'red').save(self.file, pnginfo=info)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis) as call:
            self.index.scan()
            shutil.copyfile(self.file, self.root / 'copy.png')
            self.index.scan()
            self.assertEqual(call.call_count, 1)
        for query in ('sapphire velvet', 'wardrobe', 'watermark', '123456', 'dream-model', 'DPM++'):
            self.assertEqual(self.index.search(query)['total'], 2)
        metadata = self.index.search()['items'][0]['a1111_metadata']
        self.assertEqual(metadata['raw'], text)
        self.assertEqual(metadata['negative_prompt'], 'blurry, watermark')
        self.assertTrue(metadata['parameters'].startswith('Steps: 30'))

    def test_a1111_exif_unicode_and_ascii_comments(self):
        from imageanalyser.metadata import read_a1111_metadata, decode_comment
        text = 'a café scene\nNegative prompt: fog\nSteps: 20, Seed: 42'
        exif = Image.Exif()
        exif[0x8769] = {37510: b'UNICODE\x00' + text.encode('utf-16-be')}
        jpeg = self.root / 'photo.jpg'
        Image.new('RGB', (20, 20)).save(jpeg, exif=exif)
        self.assertEqual(read_a1111_metadata(jpeg)['prompt'], 'a café scene')
        self.assertEqual(decode_comment(b'ASCII\x00\x00\x00Steps: 20'), 'Steps: 20')
        self.assertIsNone(read_a1111_metadata(self.file))

    def test_old_results_reanalysed_once_without_restoring_removed_images(self):
        from imageanalyser.core import ANALYSIS_VERSION
        second = self.root / 'second.png'
        Image.new('RGB', (40, 40), 'blue').save(second)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        self.index.remove_image(self.index.search('second')['items'][0]['id'])
        with self.index.connect() as db:
            db.execute('UPDATE images SET analysis_version=0')
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        with patch.object(restarted, 'analyse', side_effect=self.fake_analysis) as analyse:
            restarted.scan()
            restarted.scan()
            self.assertEqual(analyse.call_count, 1)
        with restarted.connect() as db:
            rows = db.execute('SELECT status,analysis_version FROM images ORDER BY id').fetchall()
            self.assertEqual(rows[0]['analysis_version'], ANALYSIS_VERSION)
            self.assertEqual(rows[1]['status'], 'removed')
        self.assertEqual(restarted.search()['total'], 1)

    def test_manual_corrections_update_search_and_survive_rescans(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        image_id = self.index.search()['items'][0]['id']
        corrected = {**self.result, 'description': 'A corrected turquoise bicycle',
                     'objects': ['turquoise bicycle'], 'people': [],
                     'a1111_raw': 'corrected prompt\nNegative prompt: old guess\nSteps: 30, Seed: 99'}
        response = client.put(f'/api/images/{image_id}', json=corrected)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(client.get('/api/search?q=turquoise').json()['total'], 1)
        self.assertEqual(client.get('/api/search?q=99').json()['total'], 1)
        self.assertTrue(client.get('/api/search').json()['items'][0]['manually_edited'])
        with patch.object(app.state.index, 'analyse') as analyse:
            app.state.index.scan()
            analyse.assert_not_called()
        self.assertEqual(client.put('/api/images/9999', json=corrected).status_code, 404)
        invalid = {**corrected, 'people': [{'age_group': '22'}]}
        self.assertEqual(client.put(f'/api/images/{image_id}', json=invalid).status_code, 422)
        self.assertTrue(self.file.exists())

    def test_manual_edit_during_captioning_is_not_overwritten(self):
        def edit_during_analysis(path):
            with self.index.connect() as db:
                image_id = db.execute('SELECT id FROM images WHERE path=?', (str(path),)).fetchone()['id']
            self.index.edit_image(image_id, {**self.result, 'description': 'Manual correction'}, '')
            return self.fake_analysis(path)
        with patch.object(self.index, 'analyse', side_effect=edit_during_analysis):
            self.index.scan()
        item = self.index.search()['items'][0]
        self.assertEqual(item['analysis']['description'], 'Manual correction')
        self.assertTrue(item['manually_edited'])

    def test_hidden_folders_are_skipped_but_hidden_filenames_are_allowed(self):
        hidden = [self.root / '.imageviewer-thumbnails' / 'nested' / 'thumb.png',
                  self.root / 'visible' / '.cache' / 'photo.png']
        allowed = [self.root / 'visible' / '.portrait.png',
                   self.root / 'imageviewer-thumbnails' / 'photo.png']
        for path in hidden + allowed:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (20, 20), 'blue').save(path)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        paths = {item['path'] for item in self.index.search()['items']}
        self.assertEqual(paths, {str(self.file), *(str(path) for path in allowed)})
        self.assertEqual(self.index.state['progress']['total'], 3)
        for path in hidden:
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):
                self.index.permitted(path)
            with patch.object(self.index, 'analyse') as analyse, patch.object(self.index.encoder, 'image') as embed:
                self.index._process_llama(path)
                self.index.process_embedding(path)
                analyse.assert_not_called()
                embed.assert_not_called()
        separate = Index({**self.config, 'roots': [str(self.root / '.imageviewer-thumbnails')],
                          'data_dir': str(self.base / 'separate-data')})
        self.addCleanup(separate.log_handler.close)
        separate.scan()
        self.assertEqual(separate.search()['total'], 0)

    def test_hidden_folder_records_purged_on_startup_and_restore(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        hidden_paths = [self.root / f'.hidden-{i}' / 'nested' / 'photo.png' for i in range(4)]
        keep = self.root / '.portrait.png'
        for path in [*hidden_paths, keep]:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (20, 20)).save(path)
        with self.index.connect() as db:
            for path, status in zip([*hidden_paths, keep], ['done', 'error', 'pending', 'removed', 'done']):
                db.execute("""INSERT INTO images(path,status,analysis,search_text,palette,sha256,analysis_version)
                    SELECT ?,?,analysis,search_text,palette,sha256,analysis_version FROM images WHERE path=?""",
                    (str(path), status, str(self.file)))
        backup = self.index.backup_database()
        self.addCleanup(backup.unlink, missing_ok=True)
        restarted = Index(self.config)
        self.addCleanup(restarted.log_handler.close)
        self.assertEqual(restarted.search()['total'], 2)
        with restarted.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM images').fetchone()[0], 2)
        restarted.restore_database(backup)
        with restarted.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM images').fetchone()[0], 2)
        self.assertTrue(all(path.exists() for path in hidden_paths))
        self.assertIn('records from hidden folders', '\n'.join(restarted.logs()))

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
        self.assertEqual(progress, {'phase': 'complete', 'scanned': 2, 'total': 2, 'discovered': 2,
                                    'previously_completed': 0})
        with patch.object(self.index, 'analyse') as analyse:
            self.index.scan()
            analyse.assert_not_called()
        self.assertEqual(self.index.state['progress']['scanned'], 2)
        self.assertEqual(self.index.state['progress']['total'], 2)

    def test_eta_uses_analysis_average_and_current_image_elapsed_time(self):
        Image.new('RGB', (30, 30), 'blue').save(self.root / 'second.png')
        Image.new('RGB', (30, 30), 'green').save(self.root / 'third.png')
        clock = [100.0]
        starts, midway = [], []
        def analyse(path):
            starts.append(self.index.progress_snapshot())
            clock[0] += 4
            midway.append(self.index.progress_snapshot())
            clock[0] += 6
            return self.fake_analysis(path)
        with patch('imageanalyser.core.time.monotonic', side_effect=lambda: clock[0]), \
             patch.object(self.index, 'analyse', side_effect=analyse):
            self.index.scan()
        self.assertIsNone(starts[0]['estimated_remaining_seconds'])
        self.assertEqual(starts[1]['average_analysis_seconds'], 10)
        self.assertEqual(starts[1]['estimated_remaining_seconds'], 20)
        self.assertEqual(midway[1]['estimated_remaining_seconds'], 16)
        self.assertEqual(starts[2]['estimated_remaining_seconds'], 10)
        complete = self.index.progress_snapshot()
        self.assertEqual(complete['analysis_count'], 3)
        self.assertEqual(complete['average_analysis_seconds'], 10)
        self.assertEqual(complete['estimated_remaining_seconds'], 0)
        # A scan that only skips existing images must not treat those as fast model calls.
        with patch.object(self.index, 'analyse') as analyse:
            self.index.scan()
            analyse.assert_not_called()
        self.assertEqual(self.index.progress_snapshot()['analysis_count'], 0)
        self.assertIsNone(self.index.progress_snapshot()['average_analysis_seconds'])

    def test_paused_scan_has_no_eta(self):
        with self.index.state_lock:
            self.index.state['progress'] = {'phase': 'paused', 'scanned': 1, 'total': 3, 'discovered': 3}
            self.index.analysis_count = 1
            self.index.analysis_seconds = 30
        self.assertIsNone(self.index.progress_snapshot()['estimated_remaining_seconds'])
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        response = TestClient(app).get('/api/status')
        self.assertIn('estimated_remaining_seconds', response.json()['progress'])

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

    def test_scan_order_changes_preserve_llama_results_and_seed_progress(self):
        from imageanalyser.settings import RuntimeSettings
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        newer = self.root / 'new.png'
        Image.new('RGB', (30, 30), 'blue').save(newer)
        self.index.save_settings(RuntimeSettings(**{**self.index.get_settings(),
                                                   'scan_sort_direction': 'desc'}))
        snapshots = []
        def analyse(path):
            snapshots.append(self.index.progress_snapshot())
            return self.fake_analysis(path)
        with patch.object(self.index, 'analyse', side_effect=analyse) as call:
            self.index.scan()
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[0], newer)
        self.assertEqual(snapshots[0]['scanned'], 1)
        self.assertEqual(snapshots[0]['total'], 2)
        self.assertEqual(snapshots[0]['previously_completed'], 1)
        self.assertEqual(self.index.search()['total'], 2)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis) as call:
            self.index.scan()
        call.assert_not_called()
        self.assertEqual(self.index.progress_snapshot()['scanned'], 2)
        # Changed content must still be analysed, even when its old caption is saved.
        Image.new('RGB', (40, 50), 'green').save(self.file)
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis) as call:
            self.index.scan()
        self.assertEqual(call.call_count, 1)
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        self.assertEqual(TestClient(app).get('/api/status').json()['llama_count'], 2)

    def test_scan_request_resumes_pause_and_avoids_duplicate_active_scan(self):
        self.index.clear()
        self.assertEqual(self.index.request_scan(), 'queued')
        self.assertFalse(self.index.paused)
        self.assertTrue(self.index.wake.is_set())
        self.index.wake.clear()
        with self.index.scan_lock:
            self.assertEqual(self.index.request_scan(), 'running')
            self.assertFalse(self.index.wake.is_set())

    def test_pause_and_cancel_during_each_analysis_stage(self):
        other = self.root / 'z-last.png'
        Image.new('RGB', (30, 30), 'blue').save(other)
        for stage in ('llama', 'openclip', 'ram'):
            for cancel in (False, True):
                with self.subTest(stage=stage, cancel=cancel):
                    config = {**self.config, 'data_dir': str(self.base / f'{stage}-{cancel}'),
                              'analysis_backend': 'openclip' if stage != 'llama' else 'llama',
                              'recognize_enabled': stage == 'ram'}
                    index = Index(config)
                    self.addCleanup(index.log_handler.close)
                    started, finish = threading.Event(), threading.Event()
                    calls = []
                    def active(path, *args):
                        calls.append(path)
                        started.set()
                        if not finish.wait(5):
                            raise RuntimeError('Test analysis timed out')
                        if stage == 'openclip':
                            return struct.pack('<ff', 1, 0)
                        if stage == 'ram':
                            return {'tags': ['person'], 'tags_chinese': []}
                        return self.fake_analysis(path)
                    target, method = ((index, 'analyse') if stage == 'llama' else
                                      (index.encoder, 'image') if stage == 'openclip' else
                                      (index.recognizer, 'image'))
                    with patch.object(index.encoder, 'image', return_value=struct.pack('<ff', 1, 0)), \
                         patch.object(target, method, side_effect=active):
                        worker = threading.Thread(target=index.scan)
                        worker.start()
                        try:
                            self.assertTrue(started.wait(5))
                            index.pause(cancel=cancel)
                            self.assertTrue(index.paused)
                            if cancel:
                                # An immediate Resume must not revive the cancelled scan/result.
                                index.resume()
                        finally:
                            finish.set()
                            worker.join(5)
                        self.assertFalse(worker.is_alive())
                    self.assertEqual(calls, [self.file])
                    field = {'llama': 'model', 'openclip': 'image_embedding', 'ram': 'ram_data'}[stage]
                    with index.connect() as db:
                        first = db.execute(f'SELECT {field} FROM images WHERE path=?', (str(self.file),)).fetchone()
                        last = db.execute(f'SELECT {field} FROM images WHERE path=?', (str(other),)).fetchone()
                    self.assertEqual(bool(first and first[0]), not cancel)
                    self.assertFalse(last and last[0])
                    if not cancel:
                        restarted = Index(config)
                        self.addCleanup(restarted.log_handler.close)
                        self.assertTrue(restarted.paused)
                        self.assertEqual(index.progress_snapshot()['phase'], 'paused')

    def test_scan_control_api_persists_pause_and_preserves_completed_images(self):
        with patch.object(self.index, 'analyse', side_effect=self.fake_analysis):
            self.index.scan()
        app = create_app(self.config)
        self.addCleanup(app.state.index.log_handler.close)
        client = TestClient(app)
        for operation in ('pause', 'cancel'):
            self.assertEqual(client.post('/api/scan/' + operation, json={'confirm': False}).status_code, 422)
            response = client.post('/api/scan/' + operation, json={'confirm': True})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['paused'])
            self.assertTrue(client.get('/api/status').json()['paused'])
            self.assertEqual(client.get('/api/search').json()['total'], 1)
            restarted = Index(self.config)
            self.addCleanup(restarted.log_handler.close)
            self.assertTrue(restarted.paused)
            self.assertEqual(client.post('/api/scan/resume', json={'confirm': True}).status_code, 200)
            self.assertFalse(app.state.index.paused)

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
