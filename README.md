# 🎓 AI Face Recognition Attendance System

> AI-based Face Recognition system for automated student attendance with hybrid verification model.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![SQLite](https://img.shields.io/badge/Database-SQLite-orange)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

##  Problem Statement

Traditional attendance systems allow students to:
- Mark attendance and leave early
- Use proxy attendance
- Exploit single-scan biometric systems

##  Solution

An AI-based Face Recognition Attendance System that:
- Verifies **continuous presence** throughout the class
- Uses **Hybrid Scanning** — Entry + Random Mid Scans + Exit
- Prevents proxy attendance using **face embeddings**
- Gives teacher full control with **manual override**

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| Face Detection | RetinaFace (InsightFace) |
| Face Recognition | Cosine Similarity (sklearn) |
| Database | SQLite + SQLAlchemy |
| Authentication | JWT |
| Camera | OpenCV (USB Webcam) |
| Frontend | HTML + TailwindCSS + JS |

---

##  Project Structure
```
Face-Detection-Attendance/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── database/
│   │   ├── db.py
│   │   └── models.py
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── enrollment_service.py
│   │   ├── recognition_service.py
│   │   ├── camera_service.py
│   │   └── attendance_service.py
│   ├── routes/
│   │   ├── auth_routes.py
│   │   ├── enrollment_routes.py
│   │   ├── attendance_routes.py
│   │   └── teacher_routes.py
│   └── middleware/
│       └── auth_middleware.py
├── frontend/
├── student_faces/
├── requirements.txt
└── README.md
```

---

##  Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Face-Detection-Attendance.git
cd Face-Detection-Attendance
```

### 2. Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Dependencies Install
```bash
pip install -r requirements.txt
```

### 4. Server Run
```bash
uvicorn backend.main:app --reload
```

### 5. Open Browser
```
API:     http://localhost:8000
Swagger: http://localhost:8000/docs
```

---

##  Attendance Logic
```
Class Start
    ↓
Entry Scan (0-10 min)
    ↓
Random Mid Scans (2-3 times)
    ↓
Exit Scan (last 10 min)
    ↓
Final Report
```

### Confidence Score System
| Score | Status |
|-------|--------|
| >= 0.75 | ✅ PRESENT |
| >= 0.50 | ⚠️ UNCERTAIN |
| < 0.50 | ❌ ABSENT |

---

##  Progress

- [x] Project Structure
- [x] Database Models
- [x] Auth System (Login/Register)
- [ ] Enrollment System
- [ ] Recognition Service
- [ ] Camera Integration
- [ ] Attendance Logic
- [ ] Teacher Dashboard
- [ ] Frontend

---
