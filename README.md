# Normexa | Unsupervised Anomaly Detection System 🧠🔍

![Platform](https://img.shields.io/badge/Platform-Web%20Dashboard-informational)
![API](https://img.shields.io/badge/API-FastAPI-0ba360)
![ML](https://img.shields.io/badge/ML-PatchCore-blue)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-646cff)
![Status](https://img.shields.io/badge/Status-Research--Oriented-success)

Normexa is an industrial-grade anomaly detection system built using PatchCore that identifies defects without requiring labeled anomaly data.

It demonstrates modern unsupervised ML engineering practices including feature extraction, memory bank modeling, nearest-neighbor anomaly scoring, FastAPI deployment, and a React + Vite interactive dashboard.

---

### ✨ Key Features

- Fully unsupervised anomaly detection (no defect labels required)
- PatchCore-based feature memory modeling
- Multi-scale feature extraction using ResNet50
- Nearest neighbor distance-based anomaly scoring
- Pixel-level anomaly localization (heatmaps)
- Multi-category support (bottle, capsule, hazelnut, metal nut, etc.)
- Risk-based classification: NORMAL / SUSPICIOUS / DEFECT
- Clean REST API integration for frontend
- Modular architecture for scalability and deployment

---

### 🧠 Problem Statement

In industrial inspection systems:

- Defects are rare and unpredictable  
- Labeled anomaly data is expensive or unavailable  
- Supervised models fail to generalize  

Traditional approaches struggle to detect unseen defects.

Normexa solves this by learning only from normal data and detecting deviations as anomalies.

---

## 🧩 System Design

### 🟢 Feature Learning & Memory Bank (Core PatchCore)

- Feature extraction using pretrained ResNet50
- Multi-layer feature fusion
- Patch-level embedding generation
- Memory bank construction using normal samples

**Answers:** What does normal look like?

---

### 🔴 Anomaly Detection & Scoring

- Compute distance between test patches and memory bank
- Nearest neighbor distance → anomaly score
- Spatial mapping → heatmap generation
- Score aggregation → image-level anomaly score

**Answers:** How different is this from normal?

---

### 🟡 Decision Layer (Risk Interpretation)

- Score normalization for interpretability
- Threshold-based classification:
  - NORMAL
  - SUSPICIOUS
  - DEFECT
- Visual explanation via heatmaps

**Answers:** Is this safe or defective?

---

## 🏗 System Architecture

React Frontend (Vite)  
↓  
FastAPI Backend  
↓  
PatchCore Inference Engine (backend/inference.py)  
↓  
Memory Bank + Model Artifacts (models/)

Architecture principles:

- separation of UI, API, and ML logic  
- reusable and modular pipeline  
- scalable for multiple categories  
- production-ready REST interface  

---

### 📦 Project Structure

CV_Proj/
│
├── backend/
│   ├── main.py
│   └── inference.py
│
├── patchcore/
│   ├── feature_extractor.py
│   ├── anomaly_scoring.py
│   ├── memory_bank.py
│   └── coreset.py
│
├── frontend/
├── models/
├── notebooks/
├── requirements.txt
└── README.md

---

## 🔌 API Specification

### POST /predict

#### Request

Form Data:
- image file
- category (e.g., bottle, capsule)

---

#### Response

{
  "score": 2.34,
  "prediction": "Defect",
  "image": "base64_encoded_heatmap"
}

---

### 🛠 Tech Stack

- Backend: Python, FastAPI, PyTorch, OpenCV
- Frontend: React, Vite
- ML: PatchCore (ResNet50 backbone)
- Deployment: REST API architecture
- Version Control: Git, GitHub

---

### ▶ Running Locally

#### Backend

pip install -r requirements.txt  
uvicorn backend.main:app --reload  

Runs at:  
http://127.0.0.1:8000  
Docs: http://127.0.0.1:8000/docs  

---

#### Frontend

cd frontend  
npm install  
npm run dev  

Runs at:  
http://localhost:5173  

---

### ⚙ Key Engineering Decisions

- unsupervised learning to handle real-world data scarcity  
- PatchCore for robust anomaly detection  
- feature normalization for stable distance computation  
- multi-scale feature fusion for better localization  
- threshold calibration using validation data  
- modular backend design for deployment readiness  

---

### 🚧 Future Improvements

- dynamic threshold learning per category  
- better feature fusion strategies  
- model optimization for real-time inference  
- cloud deployment (AWS/GCP)  
- Docker containerization  
- monitoring and logging  

---

### 📄 License

Developed for academic, research, and portfolio purposes.


