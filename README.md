# Computer Vision Smart Parking Space Occupancy & Intelligence Engine

> **AI-Powered Smart Parking Occupancy, Tracking & Predictive Analytics Platform**  
> An enterprise-grade, edge-ready, computer vision platform for real-time vehicle detection, ByteTrack spatial tracking, geometric parking-slot occupancy analysis, and XGBoost predictive demand forecasting.

---

## 🏗️ Project Architecture Overview

```
smart-parking-ai/
├── backend/            # FastAPI async REST & WebSocket server
├── cv_engine/          # OpenCV + YOLO + ByteTrack + Occupancy state machine
├── ml_engine/          # Predictive modeling (XGBoost, lag feature engineering)
├── frontend/           # React 18 + Vite + Tailwind CSS dashboard
├── database/           # PostgreSQL schemas, Alembic migrations
├── models/             # Pretrained weights (YOLO) and serialized ML estimators
├── datasets/           # Video clips, benchmark frames, annotations
├── scripts/            # Utility, verification, and benchmark scripts
├── tests/              # Pytest unit and integration test suites
├── docs/               # System documentation, API specs, diagrams
├── docker/             # Dockerfiles and deployment configurations
├── .env.example        # Environment variable template
├── .gitignore          # Version control ignore rules
├── docker-compose.yml  # Local services orchestration (Postgres, Redis)
└── README.md           # Master documentation
```

---

## 🚀 Quick Setup (Phase 1 Baseline)

### 1. Prerequisites
- Python 3.11+
- Node.js v18+ and npm
- Git
- Docker & Docker Compose (optional for local DB/Redis)

### 2. Virtual Environment Setup
```bash
# In the project root (smart-parking-ai)
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS:
source .venv/bin/activate
```

### 3. Verify Environment
Run the automated environment check script:
```bash
python scripts/verify_environment.py
```
