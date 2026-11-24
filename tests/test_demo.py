"""Exercise the demo using Streamlit's supported AppTest harness."""
import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'.demo-deps').is_dir():
    sys.path.insert(0,str(ROOT/'.demo-deps'))


@unittest.skipUnless(importlib.util.find_spec('streamlit') and (ROOT/'artifacts/portraits.sqlite3').is_file(),
                     'Demo needs Streamlit and a built gallery')
class DemoTests(unittest.TestCase):
    def app(self):
        from streamlit import config
        config.set_option('global.developmentMode',False)
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=45).run()
        self.assertFalse(app.exception)
        return app

    def test_search_exclusion_downloads_and_gallery_unchanged(self):
        with sqlite3.connect(ROOT/'artifacts/portraits.sqlite3') as db:
            before = db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
        app = self.app()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.get('download_button')),2)
        report = app.session_state['demo_result']['report']
        self.assertEqual(len(report['results']),5)
        self.assertAlmostEqual(report['results'][0]['score'],1,places=6)
        original_id = report['results'][0]['image_id']
        app.checkbox[0].check().run()
        # A changed query setting invalidates the displayed old report.
        self.assertEqual(len(app.get('download_button')),0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        report2 = app.session_state['demo_result']['report']
        self.assertEqual(report2['gallery_candidates'],report['gallery_candidates']-1)
        self.assertNotIn(original_id,[r['image_id'] for r in report2['results']])
        with sqlite3.connect(ROOT/'artifacts/portraits.sqlite3') as db:
            self.assertEqual(before,db.execute('SELECT COUNT(*) FROM images').fetchone()[0])

    def test_upload_empty_and_gallery_filter(self):
        app = self.app()
        app.radio[0].set_value('Tải ảnh lên').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)
        app.text_input[0].set_value('file-does-not-exist').run()
        self.assertTrue(any('Không tìm thấy ảnh' in info.value for info in app.info))

    def test_detection_error_is_explained(self):
        app = self.app()
        from portrait_search.features import FeatureError
        with patch('portrait_search.search.search',side_effect=FeatureError('no_face','Detected 0 faces')):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Không phát hiện khuôn mặt' in item.value for item in app.error))
        self.assertEqual(len(app.get('download_button')),0)

    def test_real_upload_and_unreadable_file(self):
        app = self.app()
        app.radio[0].set_value('Tải ảnh lên').run()
        content = (ROOT/'data_best/data_final_img_align_celeba_000055.jpg').read_bytes()
        with patch('streamlit.file_uploader',return_value=SimpleNamespace(getvalue=lambda:content)):
            app.run()
            self.assertFalse(app.button[0].disabled)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertAlmostEqual(app.session_state['demo_result']['report']['results'][0]['score'],1,places=6)
        with patch('streamlit.file_uploader',return_value=SimpleNamespace(getvalue=lambda:b'not an image')):
            app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)
        self.assertTrue(any('Không thể dùng ảnh' in item.value for item in app.error))
        self.assertEqual(len(app.get('download_button')),0)


if __name__ == '__main__':
    unittest.main()
