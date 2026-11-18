"""CLAP pretrained audio-text model: exploratory Barkopedia ZERO-SHOT evaluation.
A preliminary method, NOT an already-trained eight-class Barkopedia classifier.
"""
import argparse
import csv
from pathlib import Path
import sys

from audit_dataset import ROOT, CATEGORIES

PROMPTS = {
  'rest': 'A dog vocalizing quietly while resting.',
  'alerting to sounds': 'A dog barking to alert its owner about a sound.',
  'seeking attention': 'A dog vocalizing to seek attention from its owner.',
  'playing with human': 'A dog vocalizing while playing with a human.',
  'playing with other animals': 'A dog barking while playing with other animals.',
  'playing with toy': 'A dog vocalizing while playing with a toy.',
  'begging for food': 'A dog vocalizing while begging for food.',
  'taking shower': 'A dog vocalizing while being bathed.',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=8, help='Number of validation audio files (default 8)')
    args = ap.parse_args()
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor
    print('[INFO] CLAP is zero-shot; these similarity scores are NOT reliable behavior probabilities.')
    print('[MODEL] Loading laion/clap-htsat-unfused; first run downloads weights to LOCAL HF_HOME')
    model_id = 'laion/clap-htsat-unfused'
    model = ClapModel.from_pretrained(model_id).eval()
    proc = ClapProcessor.from_pretrained(model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[DEVICE]', device)
    model.to(device)
    manifest = ROOT/'results'/'validation_manifest.csv'
    if not manifest.exists(): raise FileNotFoundError('Run 02_AUDIT_DATA.bat first')
    with manifest.open('r', encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['audio_match']=='MATCH' and r['activity'] in CATEGORIES]
    rows = rows[:args.limit]
    if not rows: raise RuntimeError('No matched validation audio. Download and audit validation WAV first.')
    out = ROOT/'results'/'clap_zero_shot_sample.csv'
    out.parent.mkdir(exist_ok=True)
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_id', 'audio_path', 'actual_activity', 'top_1', 'top_1_softmax_over_8_descriptions', 'top_2', 'top_3', 'correct_top_1'])
        for i, item in enumerate(rows, 1):
            waveform, _ = librosa.load(ROOT/item['audio_path'], sr=48000, mono=True)
            inputs = proc(text=[PROMPTS[label] for label in CATEGORIES],
                          audios=[waveform], sampling_rate=48000, return_tensors='pt', padding=True)
            inputs = {k:v.to(device) for k,v in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits_per_audio[0]
                scores = logits.softmax(dim=0).cpu().numpy()
            picks = scores.argsort()[::-1][:3]
            names = [CATEGORIES[j] for j in picks]
            writer.writerow([item['audio_id'], item['audio_path'], item['activity'], names[0],
                             f'{float(scores[picks[0]]):.6f}', names[1], names[2], int(names[0]==item['activity'])])
            print(f'[{i}/{len(rows)}] True={item["activity"]} / CLAP={names[0]} ({float(scores[picks[0]]):.3f})', flush=True)
    print(f'[SAVED] {out}')
    print('[CAUTION] Prompt wording was not calibrated; run proper validation and train/holdout comparisons before claims.')
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}')
        sys.exit(2)
