import hashlib,io,json,tempfile,unittest,wave
from pathlib import Path
from unittest.mock import patch
import numpy as np
import live_audio as live
import test_intent_flow as t
import server
import dialogue
from engine import LABELS,evaluate_records

def wav_bytes(seconds=1,rate=16000,silent=False):
    samples=np.zeros(int(seconds*rate)) if silent else .2*np.sin(2*np.pi*500*np.arange(int(seconds*rate))/rate)
    b=io.BytesIO()
    with wave.open(b,'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes((samples*32767).astype('<i2').tobytes())
    return b.getvalue()

class AudioTests(unittest.TestCase):
    def test_native_decode_and_exact_mfcc_recipe(self):
        import librosa
        with live.decoded_audio(wav_bytes()) as (path,duration):
            mfcc,clap_wave=live.feature_waves(path)
            y,sr=librosa.load(path,sr=16000,mono=True)
            m=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13);r=librosa.feature.rms(y=y)[0];z=librosa.feature.zero_crossing_rate(y=y)[0]
            expected=np.concatenate([m.mean(1),m.std(1),[r.mean(),r.std(),z.mean(),z.std()]]).astype('float32')
            np.testing.assert_array_equal(mfcc,expected);self.assertEqual(len(clap_wave),48000);self.assertAlmostEqual(duration,1)
    def test_first_ten_seconds_for_clap_but_full_mfcc(self):
        with live.decoded_audio(wav_bytes(seconds=12)) as (path,_):
            mfcc,w=live.feature_waves(path)
            self.assertEqual(len(w),480000);self.assertEqual(mfcc.shape,(30,))
    def test_silence_short_long_invalid_rejected_and_temp_cleaned(self):
        for audio in [b'not audio',wav_bytes(.1),wav_bytes(60.2)]:
            with self.subTest(size=len(audio)),self.assertRaises(ValueError):
                with live.decoded_audio(audio):pass
        with live.decoded_audio(wav_bytes(silent=True)) as (path,_):
            saved=path
            with self.assertRaisesRegex(ValueError,'静音'):live.feature_waves(path)
        self.assertFalse(saved.exists())
    def test_playlist_and_network_reference_rejected(self):
        for raw in [b'#EXTM3U\nhttp://example.org/a.mp3',b'ffconcat version 1.0\nfile /etc/passwd']:
            with self.assertRaises(ValueError):live.media_format(raw)
    def test_format_headers(self):
        for content,fmt in [(b'ID3abc','mp3'),(b'OggSxx','ogg'),(b'fLaC','flac'),(b'\x1aE\xdf\xa3','matroska'),(b'\0\0\0\x18ftypM4A ','mov'),(b'\xff\xf1\x50','aac'),(b'\xff\xfb\x90','mp3')]:
            self.assertEqual(live.media_format(content),fmt)

class RouteTests(unittest.TestCase):
    def setUp(self):
        with server.connect() as c:c.execute('DELETE FROM sessions');c.execute('DELETE FROM revealed')
        self.row={'actual':None,'predicted':LABELS[0],'scores':dict(zip(LABELS,range(6))),'duration':2,'prediction_mode':'live','device':'cpu'}
    def test_live_runtime_orders_clap_before_mfcc(self):
        class Classifier:
            classes_=LABELS
            def predict(self,x):
                np.testing.assert_array_equal(x[0,:512],np.ones(512)*2)
                np.testing.assert_array_equal(x[0,512:],np.ones(30)*3)
                return np.array([LABELS[2]])
            def decision_function(self,x):return np.arange(6).reshape(1,6)
        runtime=live.Runtime.__new__(live.Runtime);runtime.device='cpu';runtime.classifier=Classifier()
        class Encoder:
            def vector(self,w):return np.ones(512)*2
        runtime.encoder=Encoder()
        with patch.object(live,'feature_waves',return_value=(np.ones(30)*3,np.zeros(48000))):row=runtime.predict_path(Path('fake'),1)
        self.assertIsNone(row['actual']);self.assertEqual(row['predicted'],LABELS[2]);self.assertEqual(len(row['scores']),6)

if __name__=='__main__':unittest.main()
