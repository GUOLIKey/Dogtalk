# DogTalk six-scene + E4 lightweight handoff

This branch contains the App code, the real retrained six-scene SVM, E4 emotion
weights (arousal and valence), and the latest available evaluation artifacts.
The original development directory was not modified.

Files:
- PawTalk_Web_Clean/models/scene_six_v1/live_svm.joblib
- PawTalk_Web_Clean/models/scene_six_v1/live_meta.json
- PawTalk_Web_Clean/emotion_e4_assets/arousal.joblib
- PawTalk_Web_Clean/emotion_e4_assets/valence.joblib
- docs/latest_six_class/ (when locally available)

Not bundled: raw audio, feature caches, training virtual environments, CLAP
pretrained weights, personal session histories, or DeepSeek credentials.
Dataset pointers already present on the BASE branch are not downloaded by this tool.
A teammate needs separate original datasets and the scene live-audio index for
full local inference and full training reproducibility.

Observed development-only six-scene OOF (audio >= 1 second):
Macro-F1 = 0.4270, Accuracy = 0.4659 on 4,954 DEV clips.
These are not independent test or intent-accuracy results.

Check PawTalk_Web_Clean/README_SIX_SCENES.md and README_E4.md for setup.
E4 environment: run 08_ENABLE_EMOTION.bat in the app folder.