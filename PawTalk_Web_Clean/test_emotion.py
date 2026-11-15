"""Current unchanged audio preprocessing and aggregation regression tests."""
import json,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import emotion_worker as ew
import emotion_audio as ea

class ModelContractTests(unittest.TestCase):
    def test_weighted_scores_and_gate(self):
        out=ew.aggregate([[.8,.1,.1],[.1,.1,.8]],ew.AROUSAL,[(0,2),(2,2.4)])
        self.assertAlmostEqual(out['scores']['Low'],1.64/2.4);self.assertTrue(out['usable'])
        for scores in [[[.34,.33,.33]],[[.49,.02,.49]],[[.5,.25,.25]]]:self.assertFalse(ew.aggregate(scores,ew.AROUSAL,[(0,2)])['usable'])
    def test_bad_scores_and_assets_rejected(self):
        for scores in [[[float('nan'),.1,.9]],[[1,1,1]],[[.1,.9]],[[1.1,0,-.1]]]:
            with self.assertRaises(ValueError):ew.vector_scores(scores)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):ew.verify_assets(td)
    def test_features_original_padding_pcm16_25_frames(self):
        import librosa,soundfile as sf
        from sklearn.preprocessing import StandardScaler
        wave=np.random.default_rng(10).normal(0,.08,60000).astype(np.float32);scaler=StandardScaler().fit(np.arange(50).reshape(-1,1))
        actual,spans=ew.features(wave,scaler);expected=[]
        for start in range(0,len(wave),44100):
            chunk=wave[start:start+44100]
            if len(chunk)<44100:chunk=np.concatenate([chunk,np.zeros(44100-len(chunk))])
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'chunk.wav';sf.write(p,chunk,22050);y,sr=librosa.load(p,sr=22050)
                m=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=1,n_fft=2048,hop_length=512).T
                expected.append(scaler.transform(m[:25]))
        np.testing.assert_allclose(actual,np.asarray(expected,dtype=np.float32),rtol=0,atol=0)
        self.assertEqual(actual.shape,(2,25,1));self.assertAlmostEqual(spans[-1][1],len(wave)/22050)
    def test_missing_setup_preserves_context(self):
        with patch.object(ea,'status',return_value={'ready':False,'message':'run 08'}):out=ea.predict_bytes(b'audio')
        self.assertFalse(out['available']);self.assertNotIn('arousal',out)
    def test_timeout_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as td:
            config=Path(td)/'runtime.json';config.write_text(json.dumps({'python':'test-python','recipe':'test'}));paths=[]
            def timeout(args,**kwargs):
                paths.append(Path(args[args.index('--audio')+1]));self.assertTrue(paths[0].exists());raise subprocess.TimeoutExpired(args,1)
            with patch.object(ea,'RUNTIME',config),patch.object(ea,'status',return_value={'ready':True}),patch.object(ea.subprocess,'run',side_effect=timeout):out=ea.predict_bytes(b'audio')
            self.assertFalse(out['available']);self.assertFalse(paths[0].exists())

if __name__=="__main__":unittest.main()
