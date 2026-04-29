# ARKK — The College Assistant Humanoid Robot

[![IEEE ICCR 2024](https://img.shields.io/badge/IEEE%20ICCR-2024-blue)](https://doi.org/10.1109/ICCR64365.2024.10927589)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Nano-orange)](https://developer.nvidia.com/embedded/jetson-nano)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Published at the **6th IEEE International Conference on Control and Robotics (ICCR 2024)**, Yokohama, Japan.  
> DOI: [10.1109/ICCR64365.2024.10927589](https://doi.org/10.1109/ICCR64365.2024.10927589)

---

## Overview

ARKK is a humanoid college assistant robot designed to redefine the visitor and student experience at educational institutions. It integrates real-time face detection, face recognition, multi-modal emotion recognition, an NLP-powered chatbot (RASA), library assistance, campus navigation, staff attendance monitoring, and automated mailing — all running on an NVIDIA Jetson Nano.

---

## Features

| Feature | Description |
|---|---|
| **Face Detection & Recognition** | Haar Cascade-based real-time identification of known/unknown individuals |
| **Face Emotion Recognition** | VGG-16 CNN trained on FER-2013 (73.2% training accuracy) |
| **Speech Emotion Recognition** | Custom 7-layer CNN on CREMA-D, RAVDESS, SAVEE, TESS (98.12% accuracy) |
| **NLP Chatbot (RASA)** | Handles campus queries: admissions, staff, library, navigation, clubs, etc. |
| **Campus Navigation** | Floor-map display with spoken directions to any room/office |
| **Library Assistance** | Book availability lookup by title or author |
| **Staff Attendance** | Continuous face-based attendance logged to a spreadsheet |
| **Mailing** | Automated appointment/request emails to college administration |
| **GPT / Gemini Integration** | Fallback general-knowledge responses via Google Gemini API |

---

## Hardware Architecture

```
┌──────────────────────────────────────────────────┐
│                  Jetson Nano (Brain)              │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Camera  │  │   Mic    │  │  Touch Display   │ │
│  │ (USB)   │  │  (USB)   │  │    (HDMI)        │ │
│  └─────────┘  └──────────┘  └──────────────────┘ │
│  ┌──────────────────────┐   ┌──────────────────┐ │
│  │    Arduino Nano      │   │    Speaker (USB) │ │
│  │  (Head & Jaw Servos) │   └──────────────────┘ │
│  └──────────────────────┘                        │
└──────────────────────────────────────────────────┘
```

- **Jetson Nano** — Central processing (face recognition, emotion detection, NLP, speech)
- **Arduino Nano** — Servo control for head and jaw movements
- **3D-Printed Head** — ABS thermoplastic, CAD-designed, servo-actuated
- **Female Mannequin Body** — Fiber construction housing all electronics
- **Touch Screen Display** — Body-mounted for touch-based input/output

---

## Software Architecture

```
Live Video Feed ──► Face Detection ──► Face Recognition ──► Screen
                         │
                         └──► Emotion Recognition ──► Speech Synthesis ──► Speaker
                                       │
Speech Input ──────────────────────────┘
                         │
User Input Query ──────► RASA Chatbot ──► Response
                              │
                    (fallback) └──► Gemini API ──► Response
```

### Model Details

**Face Emotion Detection (VGG-16)**
- Dataset: FER-2013 (35,887 images, 7 emotion classes)
- Architecture: 13 Conv layers + 3 Fully Connected layers
- Final validation accuracy: **60.67%** (50 epochs)

**Speech Emotion Recognition (Custom CNN)**
- Datasets: CREMA-D, RAVDESS, SAVEE, TESS
- Architecture: 5 Conv layers + 2 Dense layers + 6 Batch Norm layers
- Accuracy: **98.12%** (6 emotion classes)

---

## Repository Structure

```
ARKK/
├── src/
│   ├── app.py                        # Main application (final, no Gemini)
│   ├── main.py                       # Core face + emotion + chatbot loop
│   ├── main2.py                      # Variant with extended features
│   ├── mainStage2.py                 # Stage 2 development version
│   ├── mainst2.py                    # Stage 2 alternate
│   ├── stage2_with_gemini.py         # Gemini-integrated version
│   ├── stage3_with_gemini_tkinter.py # Tkinter GUI + Gemini version
│   ├── check_voice.py                # Voice engine test utility
│   ├── haarcascade_frontalface_default.xml
│   ├── library_book_database_final2.csv
│   ├── trainer/
│   │   └── trainer.yml               # LBPH face recognizer model
│   └── FaceEmotionDetection/
│       └── haarcascade_frontalface_default.xml
├── models/
│   └── emotion_model_1.h5            # Trained face emotion CNN (place here)
├── rasa_chatbot/                     # RASA NLU + dialogue files (see below)
├── assets/
│   └── navigation_maps/              # Floor map images (a.jpg–q.jpg)
├── docs/
│   └── ARKK_paper_ICCR2024.pdf       # Published IEEE paper
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- NVIDIA Jetson Nano (or any Linux machine with Python 3.8+)
- USB Camera, Microphone, Speaker
- RASA server running locally on `http://localhost:5005`

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ARKK.git
cd ARKK
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Place model files

- Copy `emotion_model_1.h5` into `models/`
- Copy navigation map images (`a.jpg` – `q.jpg`) into `assets/navigation_maps/`
- Copy `bg.png` (UI background) into `src/`

### 4. Start the RASA server

```bash
cd rasa_chatbot
rasa run --enable-api
```

### 5. Run the application

```bash
cd src
python app.py
```

> Use `main.py` for the minimal command-line version without the Tkinter GUI.

---

## RASA Chatbot

The chatbot is trained to respond to campus-specific intents including:

- Admission procedures
- Staff availability (via live face-based attendance)
- Library book search (`library_book_database_final2.csv`)
- Campus navigation (room/office directions with map display)
- Exam and placement information
- Club and society queries
- Appointment booking and mailing

Place your RASA project files (`domain.yml`, `config.yml`, `nlu.yml`, `stories.yml`, `rules.yml`) in the `rasa_chatbot/` directory.

---

## Navigation Maps

The `stage4.py` / `app.py` navigation system maps room names to floor images:

```python
d = {
    'Principal Room': 'd.jpg',
    'Library': 'i.jpg',
    'HOD Mechanical': 'e.jpg',
    ...
}
```

Place corresponding floor-plan images in `assets/navigation_maps/` and update the path in `app.py` accordingly.

---

## Results Summary

| Module | Metric | Value |
|---|---|---|
| Face Emotion Recognition | Training Accuracy | 73.20% |
| Face Emotion Recognition | Validation Accuracy | 60.67% |
| Speech Emotion Recognition | Accuracy | **98.12%** |
| Face Recognition | Method | LBPH (Haar Cascade) |
| Chatbot | Framework | RASA Open Source |

---

## Publication

If you use this work, please cite:

```bibtex
@inproceedings{ananthan2024arkk,
  author    = {Ananthapadmanabhan, M. and Rahul, R. and Karthiyayani Menon, R. and Krishna, R. Jain and Hari, R.},
  title     = {ARKK: The College Assistant Humanoid Robot},
  booktitle = {2024 6th International Conference on Control and Robotics (ICCR)},
  year      = {2024},
  pages     = {234--242},
  doi       = {10.1109/ICCR64365.2024.10927589},
  publisher = {IEEE}
}
```

---

## Authors

- **Ananthapadmanabhan M.** — [iamananthan23@gmail.com](mailto:iamananthan23@gmail.com)
- Rahul R.
- R. Karthiyayani Menon
- Krishna R. Jain
- Hari R. *(Faculty Advisor)*

Government Engineering College Barton Hill, Thiruvananthapuram  
APJ Abdul Kalam Technological University, Kerala, India

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
