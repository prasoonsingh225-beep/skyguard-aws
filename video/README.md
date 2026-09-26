# AERO Guard SIH 26073 Video Package

This folder contains a ready-to-render 3-minute SIH demonstration video package for **AERO Guard — Intelligent Anomaly Detection for Automatic Weather Stations**, based on SIH Problem Statement 26073.

## Fastest way to render

Install FFmpeg, then run:

```bash
cd video
bash build_video.sh
```

This creates `aero_guard_sih_26073.mp4` using the included storyboard slides and narration script. If `gtts` is installed and internet is available, the script generates narration automatically. Otherwise it creates a professional silent-subtitle video that can be narrated in OBS or PowerPoint.

Optional narration setup:

```bash
python -m pip install gTTS
bash build_video.sh
```

## What the video demonstrates

1. The AWS data-quality problem and its operational impact
2. AERO Guard's multivariate AI pipeline
3. Synthetic data and injected anomaly testing
4. LSTM autoencoder scoring
5. FastAPI real-time response with severity and confidence
6. Streamlit monitoring dashboard
7. Explainable root-cause reasoning
8. ESP32/TFLite edge deployment path
9. SIH 26073 alignment and future self-healing network vision

## Recommended recording

For the strongest final submission, replace `demo_terminal.svg` and `demo_dashboard.svg` with screenshots or 10–15 second screen recordings of the running prototype. The video still renders without those replacements.
