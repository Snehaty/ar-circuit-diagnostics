# AR-Based Intelligent Circuit Diagnostics System

An AI-powered system that detects electronic circuit components in real-time using Computer Vision, creates a Digital Twin, diagnoses faults, and displays AR overlays with repair guidance.

## Tech Stack
- **Computer Vision:** Python, OpenCV, YOLOv8
- **ML/AI:** PyTorch, Transfer Learning
- **AR Layer:** Unity, Vuforia Engine
- **API:** FastAPI
- **Generative AI:** Google Gemini API

## Features
- ✅ Real-time component detection (18 classes)
- 🔄 Circuit reconstruction & Digital Twin
- 🔍 Fault detection using ML
- 🔧 AI-powered repair guidance
- 📱 AR overlay on physical circuits

## Components Detected
Battery, Capacitor, Diode, Fuse, IC, Inductor, LDR, LED, Photodiode, Potentiometer, Resistor, Speaker, Switch, Thermistor, Transformer, Transistor, Trimpot, Varco

## Project Structure
ar-circuit-diagnostics/

├── cv_pipeline/      # YOLO detection scripts

├── fault_detection/  # ML fault classifier

├── api/              # FastAPI server

├── datasets/         # Training data (gitignored)

└── models/           # Trained weights (gitignored)
## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python fastapi uvicorn streamlit
```

## Run Detection
```bash
cd cv_pipeline
python3 detect.py
```
