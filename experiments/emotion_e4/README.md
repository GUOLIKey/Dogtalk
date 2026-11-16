# E4 final deployment refit

This directory reproduces the selected emotion-only E4 procedure. Scene models,
scene caches, label definitions and 24 intent rules are unchanged.

## Data and evaluation boundary

`dataset_manifest.json` pins the upstream EmotionalCanines revision, official
TRAIN file sizes and SHA256 values. Only the 1,200 TRAIN recordings and their
arousal/valence annotations are downloaded. Breed is **not** the emotion target.
An existing local dataset can be supplied with `--dataset-dir`; its bytes still
have to match the manifest. Git LFS pointers are rejected as invalid audio.

`reference/` contains the user's existing E4 development results, exact fold
assignments, selected parameters and training fingerprints. The script verifies
the dataset and grouping, reproduces the recorded outer-fold Macro-F1, then
selects parameters by three-fold grouped CV on the full TRAIN data and refits.
The selection grid is unchanged: scaled logistic regression and RBF SVM,
C=0.1/1/10, SVM gamma=scale/0.01/0.1, class_weight=None/balanced.

The full-data CV selection score is **not** an independent evaluation. The
reported E4 OOF metrics remain development comparison results after selecting
E4. Official TEST is not decoded, fitted or scored by this script. Previously
viewed test data cannot subsequently be called a fresh blind test.
Dog IDs and source-recording IDs are unavailable: 1,198 fingerprint groups do
not establish generalization to unseen dogs. Conflicting duplicate labels are
retained and kept in the same fold, rather than removed to improve scores.

## Reproduce (Python 3.12 recommended)

From this directory, in a dedicated environment:

```powershell
python -m pip install -r requirements.txt
python train_final.py
# Or reuse verified local training recordings:
python train_final.py --dataset-dir "D:\your\EmotionalCanines"
```

Default output is `../../PawTalk_Web_Clean/emotion_e4_assets/`, containing two
final joblib pipelines and `metadata.json`. `emotion_e4_spec.json` records their
hashes. If final model-family selection does not choose SVM, publication stops
for review instead of silently mislabelling a different model as this release.

After a deliberate rebuild, regenerate the release checksum manifest:

```powershell
cd ../../PawTalk_Web_Clean
python build_integrity.py
python start_app.py --check
```

Rebuilding replaces only E4 artifacts. The app never retrains in response to a
user answer. The shared `emotion_e4_features.py` defines all 153 features; the
training metadata pins its source hash and numerical-library versions.
