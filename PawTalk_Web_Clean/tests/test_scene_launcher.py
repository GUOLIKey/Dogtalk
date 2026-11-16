"""Launcher/report regression tests. Mocked training is never a real score claim."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_scene_experiment as runner
import scene_results_view as view

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.metrics=json.loads((view.RESULTS/'historical_audit'/'metrics.json').read_text())
    def result(self,path,synthetic=False):
        path.mkdir(parents=True,exist_ok=True);data=copy.deepcopy(self.metrics)
        if synthetic:data['provenance']={'kind':'paired_model_rerun','data_kind':'synthetic_test_fixture'}
        (path/'metrics.json').write_text(json.dumps(data),encoding='utf-8');return path
    def test_historical_report_never_claims_retraining(self):
        path=self.result(self.root/'historical_audit');report,text=view.render(path)
        self.assertIn('真正六分类重训尚未完成',text)
        self.assertIn('重新训练六分类模型（新任务）: Accuracy=未运行, Macro-F1=未运行',text)
        self.assertIn('Macro-F1=0.4243',text);self.assertIn('混淆矩阵',report.read_text())
    def test_synthetic_report_is_explicit(self):
        report,text=view.render(self.result(self.root/'test',synthetic=True))
        self.assertIn('仅为合成测试数据',text);self.assertNotIn('真实 DEV 同折比较已完成',text)
    def test_incomplete_training_not_selected_as_latest(self):
        old=self.result(self.root/'historical_audit');self.result(self.root/'trained_run_incomplete')
        self.assertEqual(view.select_result(self.root),old)
        done=self.result(self.root/'trained_run_complete');(done/'RUN_COMPLETE.json').write_text('{}')
        self.assertEqual(view.select_result(self.root),done)
    def test_missing_resources_fail_before_expensive_work(self):
        manifest=self.root/'dev.csv';manifest.write_text('audio_path,actual\nmissing.wav,rest\n')
        with self.assertRaisesRegex(RuntimeError,'缺少 1 条 DEV'):
            runner.check_resources(self.root,manifest,self.root/'holdout.csv')
    def test_lock_rejects_duplicate_and_releases(self):
        lock=self.root/'training.lock'
        with runner.experiment_lock(lock):
            with self.assertRaises(RuntimeError):
                with runner.experiment_lock(lock):pass
        with runner.experiment_lock(lock):pass
    def test_cache_export_failure_never_silently_switches_features(self):
        app=self.root/'PawTalk_Web_Clean';(app/'models').mkdir(parents=True);(app/'models'/'live_meta.json').write_text('{}')
        scripts=self.root/'scripts';scripts.mkdir()
        for name in ('step4b1_denoise_abc.py','step3_svm_tuning.py'):(scripts/name).write_text('#fixture')
        a=argparse.Namespace(project_root=self.root,holdout_manifest=None,fresh=False,no_install=True,no_browser=True)
        with patch.object(runner,'HERE',app),patch.object(runner,'check_resources',return_value=9984),patch.object(runner,'export_existing',side_effect=ValueError('parity mismatch')),patch.object(runner,'extract') as extract,patch.object(runner,'train') as train:
            with self.assertRaisesRegex(RuntimeError,'原缓存导出失败'):runner.run(a)
        extract.assert_not_called();train.assert_not_called()
    def test_cached_package_is_used_and_mock_run_cannot_be_published_as_real(self):
        app=self.root/'PawTalk_Web_Clean';(app/'data').mkdir(parents=True)
        for name in ('scene_features_v1.npz','scene_features_v1.npz.meta.json'):(app/'data'/name).write_text('fixture')
        a=argparse.Namespace(project_root=self.root,holdout_manifest=None,fresh=False,no_install=True,no_browser=True)
        def fit(args):self.result(args.output,synthetic=True)
        with patch.object(runner,'HERE',app),patch.object(runner,'check_resources',return_value=9984),patch.object(runner,'train',side_effect=fit) as train,patch.object(runner,'extract') as extract,patch.object(runner,'export_existing') as export:
            with self.assertRaisesRegex(ValueError,'不能把合成训练标为真实完成'):runner.run(a)
        train.assert_called_once();extract.assert_not_called();export.assert_not_called()
        self.assertFalse(list(app.rglob('RUN_COMPLETE.json')))

if __name__=='__main__':unittest.main()
