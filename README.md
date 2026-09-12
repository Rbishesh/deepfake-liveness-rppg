# Deepfake-Proof Live Identity Verification (rPPG Liveness Detection)

A real-time biometric and physiological liveness verification system designed to detect generative AI deepfakes, face swaps, and video replay attacks using **Remote Photoplethysmography (rPPG)**.

---

## 🌟 The Core Innovation

Generative AI (FaceSwap, DeepFaceLive, LivePortrait, Diffusion models) can produce photorealistic surface pixels, but **cannot simulate human cardiovascular hemodynamics**. 

When a living heart beats, blood is pumped through capillary micro-vessels in facial tissue. Hemoglobin absorbs green-spectrum light, causing periodic, imperceptible fluctuations (0.5% – 1.5%) in facial skin color at **0.75 Hz – 2.5 Hz (45 – 150 BPM)**.

### Why Deepfakes Fail
1. **Cardiac SNR:** Real human skin displays a sharp, concentrated spectral peak at the heart rate frequency. Deepfakes or photos exhibit diffuse broadband noise or a flatline.
2. **Multi-Region Cross-Correlation:** In living tissue, the forehead, left cheek, and right cheek pulse in near-perfect synchrony ($r > 0.7$). Face-swap models generate regions asynchronously or patch by patch, breaking spatial coherence.

---

## 🚀 Quick Start

### 1. Requirements
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### 2. Run with Live Webcam
```bash
python main.py
```
Or specify camera index:
```bash
python main.py --source 0
```

### 3. Run with a Video File (e.g. Deepfake vs Real sample)
```bash
python main.py --source path/to/video.mp4
```

### 4. Run the Synthetic Demo Simulator (No Camera Required)
```bash
python main.py --synthetic-demo
```
This launches an interactive simulation alternating between a genuine human cardiac pulse (72 BPM) and an asynchronous deepfake spoof.

---

## 🎮 Interactive Controls
- `q` or `ESC`: Quit application.
- `s`: Save a high-resolution verification report snapshot (`liveness_report_<timestamp>.png`).
- `r`: Reset buffer to recalibrate.

---

## 📂 Project Structure

```
deepfake-liveness-rppg/
├── data/
│   └── models/
│       └── face_landmarker.task   # MediaPipe FaceLandmarker task (auto-downloaded)
├── src/
│   ├── core/
│   │   ├── face_tracker.py       # Multi-region ROI tracking (forehead + cheeks)
│   │   ├── rppg_extractor.py     # POS, CHROM, and Green-channel algorithms
│   │   ├── signal_processing.py  # Butterworth filter, FFT, PSD, and SNR
│   │   └── liveness_verifier.py  # Multi-region sync & anti-spoofing engine
│   └── ui/
│       └── dashboard.py          # Real-time HUD and oscilloscope waveform graph
├── tests/
│   ├── test_pipeline.py          # Mathematical & filter unit tests
│   └── test_end_to_end.py        # End-to-end simulated verification tests
├── requirements.txt              # Project dependencies
├── main.py                       # CLI & real-time dashboard runner
└── README.md                     # Documentation
```

---

## 🧪 Testing the Algorithms
Run the automated test suite:
```bash
python -m unittest discover tests
```
