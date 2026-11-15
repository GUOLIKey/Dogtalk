import hashlib,json,tempfile,unittest
from pathlib import Path
from install_update import install
class Installer(unittest.TestCase):
 def test_preserves_private_assets_and_backs_up_code(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);src=root/'source';dst=root/'target';src.mkdir();dst.mkdir()
   (src/'server.py').write_text('new');(dst/'server.py').write_text('old');(dst/'00_START_APP.bat').write_text('start')
   for name in ['DEEPSEEK_CONFIG.ini','data/a','models/a','state/a','.venv_emotion/a']:
    f=dst/name;f.parent.mkdir(exist_ok=True);f.write_text('private')
   (src/'build_manifest.json').write_text(json.dumps({'sha256':{'server.py':hashlib.sha256(b'new').hexdigest()}}))
   backup=install(src,dst);self.assertEqual((dst/'server.py').read_text(),'new');self.assertEqual((backup/'server.py').read_text(),'old')
   for name in ['DEEPSEEK_CONFIG.ini','data/a','models/a','state/a','.venv_emotion/a']:self.assertEqual((dst/name).read_text(),'private')
 def test_rejects_bad_package_before_changes(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);src=root/'source';dst=root/'target';src.mkdir();dst.mkdir()
   (src/'server.py').write_text('new');(dst/'server.py').write_text('old');(dst/'00_START_APP.bat').write_text('start')
   (src/'build_manifest.json').write_text(json.dumps({'sha256':{'server.py':'bad'}}))
   with self.assertRaises(ValueError):install(src,dst)
   self.assertEqual((dst/'server.py').read_text(),'old')
