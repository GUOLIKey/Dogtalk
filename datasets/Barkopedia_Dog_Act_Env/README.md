---
license: mit
task_categories:
- audio-classification
language:
- en
tags:
- biology
- dog
pretty_name: BpAE
size_categories:
- 10K<n<100K
---

# 🐶 Barkopedia Challenge Dataset  
[🔗 Barkopedia Website](https://uta-acl2.github.io/barkopedia.html)

## 📦 Dataset Description

This challenge provides a labeled dataset of dog bark audio clips for understanding **activity** and **environment** from sound.

### 📁 Current Release

- **Training Set**
  - Located in the `train/` folder
  - Includes:
    - `split1.zip` and `split2.zip` — each contains a portion of the audio files
    - `train_label.csv` — contains labels for all training clips
  - Total: **12,480** training audio clips

- **Test Set**
  - To be released in **June**
  - Will contain **3,120** audio clips
    - **40% public (1,248)** — used for live leaderboard updates
    - **60% private (1,872)** — used for final evaluation

## 🔖 Labels

Each clip is annotated with **one activity** and **one environment** category.

### 🎯 Activity Categories (`act_category`)
- `rest`  
- `alerting to sounds`  
- `seeking attention`  
- `playing with human`  
- `playing with other animals`  
- `playing with toy`  
- `begging for food`  
- `taking shower`  

### 🌍 Environment Categories (`env_category`)
- `indoor (general)`  
- `near window`  
- `near door`  
- `on grass`  
- `near other animals`  
- `vehicle interior`  

> Labels were initially generated using video-assisted inference via a visual-language model (Janus-Pro-7B), and later manually verified to ensure quality.

## 🚀 Submission & Leaderboard

All competition data, result submission, and the live leaderboard are hosted on **[Hugging Face](https://huggingface.co/)**.  
If you don’t have a Hugging Face account yet, please [register here](https://huggingface.co/join).

---

📌 *Stay tuned for the test set release in June and leaderboard launch!*

---