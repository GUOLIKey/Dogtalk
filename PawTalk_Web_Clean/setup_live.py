"""Prepare frozen CLAP only after a genuine six-class SVM is installed."""
from live_audio import status, Runtime, Encoder

def main():
    state=status()
    if not state['ready']:raise ValueError(state['message']+'。见 README_SIX_SCENES.md；07 不会把旧模型改名。')
    encoder=Encoder(allow_download=True)
    del encoder
    runtime=Runtime()
    print('READY: '+state['model_version']+'; device='+runtime.device)

if __name__=='__main__':
    try:main()
    except Exception as exc:print('[NOT READY] '+str(exc));raise SystemExit(1)
