"""Frozen E4 rich feature recipe, shared by deployment and final fitting."""
RECIPE='full_mfcc20_delta_spectral_rhythm_v1'
import numpy as np

def extract_vector(wave,sr=22050):
    import librosa
    if len(wave)<2048:wave=np.pad(wave,(0,2048-len(wave)))
    m=librosa.feature.mfcc(y=wave,sr=sr,n_mfcc=20,n_fft=2048,hop_length=512)
    basic=np.r_[m[:13].mean(axis=1),m[:13].std(axis=1)]
    delta=librosa.feature.delta(m,width=3,mode='nearest')
    rich=[m.mean(axis=1),m.std(axis=1),np.quantile(m,.1,axis=1),np.quantile(m,.9,axis=1),delta.mean(axis=1),delta.std(axis=1)]
    S=np.abs(librosa.stft(wave,n_fft=2048,hop_length=512))
    tracks=[librosa.feature.rms(S=S)[0],librosa.feature.zero_crossing_rate(wave,frame_length=2048,hop_length=512)[0],
      librosa.feature.spectral_centroid(S=S,sr=sr)[0],librosa.feature.spectral_bandwidth(S=S,sr=sr)[0],
      librosa.feature.spectral_flatness(S=S)[0],librosa.feature.spectral_rolloff(S=S,sr=sr)[0]]
    for t in tracks:rich.append(np.array([t.mean(),t.std(),*np.quantile(t,[.1,.5,.9])]))
    rms=tracks[0];active=rms>max(1e-5,float(np.median(rms))*.5)
    rich.append(np.array([active.mean(),np.mean(np.abs(np.diff(rms))) if len(rms)>1 else 0.,np.mean(active[1:]!=active[:-1]) if len(active)>1 else 0.]))
    rich=np.concatenate(rich).astype(np.float32)
    if not np.isfinite(rich).all():raise ValueError('非有限特征')
    return basic.astype(np.float32),rich

def read_features(path):
    import librosa,soundfile as sf
    native,sr=sf.read(str(path),dtype='float32',always_2d=True)
    if native.size==0 or not np.isfinite(native).all():raise ValueError('录音为空或采样无效。')
    wave=librosa.to_mono(native.T)
    duration=len(wave)/sr
    if float(np.sqrt(np.mean(wave.astype(float)**2)))<1e-5:raise ValueError('录音接近静音。')
    if sr!=22050:wave=librosa.resample(wave,orig_sr=sr,target_sr=22050)
    _,rich=extract_vector(wave)
    if rich.shape!=(153,):raise ValueError('E4特征维度错误。')
    return rich,duration
