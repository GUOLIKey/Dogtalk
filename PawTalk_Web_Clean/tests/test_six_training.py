"""Actual SVM fits on synthetic features/audio, NOT a Barkopedia performance test."""
import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import joblib
import scene_training as train
from scene_schema import LABELS, ORIGINAL_LABELS, MODEL_VERSION, LABEL_VERSION
from scene_cache import load_index

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.rows=[];rng=np.random.default_rng(11)
        for c,label in enumerate(ORIGINAL_LABELS):
            for n in range(6):
                i=c*6+n;path=self.root/f'{i}.wav'
                sf.write(path,rng.normal(0,.1,800).astype(np.float32),8000)
                self.rows.append(dict(audio_path=path.name,actual=label,A_pred=label,fold=n%3+1,duration_sec=.1,eligible_1s=0))
        # Same waveform across originally different folds must trigger corrected grouping.
        (self.root/'1.wav').write_bytes((self.root/'0.wav').read_bytes())
        self.manifest=self.root/'dev.csv';train.write_csv(self.manifest,self.rows)
        self.holdout=[{'audio_path':'locked.wav','actual':'rest'}]
        self.hold_path=self.root/'holdout.csv';train.write_csv(self.hold_path,self.holdout)
        self.features=self.root/'features.npz'
        np.savez_compressed(self.features,clap=rng.normal(size=(48,512)).astype(np.float32),mfcc=rng.normal(size=(48,30)).astype(np.float32),audio_paths=np.array([r['audio_path'] for r in self.rows]),feature_version=train.FEATURE_VERSION)
        train.write_json(str(self.features)+'.meta.json',dict(sha256=train.sha(self.features),feature_version=train.FEATURE_VERSION,manifest_sha256=train.sha(self.manifest),librosa_version='fixture',transformers_version='fixture',data_kind='synthetic_test_fixture'))
    def args(self):
        return argparse.Namespace(manifest=self.manifest,project_root=self.root,holdout_manifest=self.hold_path,features=self.features,output=self.root/'out',install=False)
    def test_full_paired_cv_train_and_cache(self):
        a=self.args();train.train(a)
        result=json.loads((a.output/'metrics.json').read_text())
        self.assertEqual(set(result['subsets']['all_dev']),{'original_eight','mapped_eight_to_six','retrained_six'})
        self.assertEqual(result['provenance']['data_kind'],'synthetic_test_fixture')
        self.assertTrue(result['provenance']['same_samples_and_folds'])
        audit=result['provenance']['split_audit'];self.assertEqual(audit['original_cross_fold_groups'],1)
        self.assertTrue(audit['split_policy'].startswith('corrected_'))
        folds=train.read_csv(a.output/'fold_assignments.csv');self.assertEqual(folds[0]['fold'],folds[1]['fold'])
        model=joblib.load(a.output/'live_svm.joblib');self.assertEqual(set(model.classes_),set(LABELS));self.assertEqual(model.n_features_in_,542)
        cache=load_index(a.output/'audio_index.json');self.assertEqual(len(cache['items']),46)
        self.assertEqual(cache['model_version'],MODEL_VERSION);self.assertEqual(cache['label_version'],LABEL_VERSION)
        for r in cache['items'].values():self.assertEqual(set(r['scores']),set(LABELS))
        self.assertFalse(audit['holdout_audio_decoded_or_scored'])
        with self.assertRaisesRegex(ValueError,'非空'):train.train(a)
    def test_locked_holdout_identity_blocks(self):
        with self.assertRaisesRegex(ValueError,'路径重叠'):train.audit_groups(self.rows,self.root,[{'audio_path':'0.wav'}])
        with self.assertRaisesRegex(ValueError,'指纹重叠'):train.audit_groups(self.rows,self.root,[dict(audio_path='different.wav',audio_sha256=train.sha(self.root/'0.wav'))])
    def test_dog_and_source_groups_are_transitive(self):
        self.rows[3]['dog_id']='dog1';self.rows[4]['dog_id']='dog1'
        self.rows[4]['source_recording_id']='recording1';self.rows[5]['source_recording_id']='recording1'
        _,groups,folds,audit=train.audit_groups(self.rows,self.root,self.holdout)
        self.assertEqual(groups[3],groups[5]);self.assertEqual(folds[3],folds[5]);self.assertEqual(audit['dog_id_coverage'],2)
    def test_synthetic_training_cannot_be_installed(self):
        a=self.args();a.install=True
        with self.assertRaisesRegex(ValueError,'禁止安装模拟'):train.train(a)
    def test_bad_feature_order_stops_before_training(self):
        with np.load(self.features) as z:values={k:z[k] for k in z.files}
        values['audio_paths']=values['audio_paths'][::-1];np.savez_compressed(self.features,**values)
        path=Path(str(self.features)+'.meta.json');m=json.loads(path.read_text());m['sha256']=train.sha(self.features);train.write_json(path,m)
        with self.assertRaisesRegex(ValueError,'特征顺序'):train.train(self.args())

if __name__=='__main__':unittest.main()
