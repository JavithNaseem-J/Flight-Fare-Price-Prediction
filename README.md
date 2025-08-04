# ✈️ Flight Fare Finder

End-to-End Machine Learning Pipeline 🚀

---

![Flight-Fare-MLops-Image](https://github.com/user-attachments/assets/588a03d2-8ac2-49da-ba66-068ba90a8caf)

---

## 📚 Project Overview

This project is a **full MLOps-grade system** that predicts flight fare prices based on flight details such as airline, route, booking class, and time to departure.

- ⛓️ Modular pipeline stages
- 🧪 ML experiment tracking with MLflow
- 📦 Data & model versioning with DVC
- 🚀 REST API with FastAPI
- 🐳 Docker containerization
- ⚙️ CI/CD automation via GitHub Actions & AWS ECR

---

## 🏧 Project Architecture

```
Data Ingestion → Data Validation → Data Cleaning →
Data Transformation → Model Training → Model Evaluation →
Prediction API →Docker Containerization → CI/CD → AWS ECR Deployment
```

---

## 🚀 Tech Stack

| Category        | Tools Used                                |
| --------------- | ----------------------------------------- |
| Language        | Python 3.10                               |
| ML Framework    | Scikit-learn, XGBoost                     |
| Experimentation | MLflow + Dagshub                          |
| Versioning      | DVC                                       |
| Deployment      | FastAPI + Uvicorn                         |
| Packaging       | Docker                                    |
| Automation      | GitHub Actions → AWS ECR → EC2 Deployment |

---

## 📂 Folder Structure

```
├── config/
├── src/mlproject/
│   ├── components/
│   ├── pipelines/
│   ├── config/
│   ├── entities/
│   └── utils/
├── artifacts/
├── dvc.yaml
├── params.yaml
├── schema.yaml
├── Dockerfile
├── app.py
├── README.md
├── requirements.txt
└── .github/workflows/cicd.yml
```

---

## 💠 Setup Instructions

### 🔹 Clone the repository

```bash
git clone https://github.com/JavithNaseem-J/FareFinder.git
cd FareFinder
```

### 🔹 Create & activate virtual environment

```bash
conda create <env name> python=3.10 -y
conda activate <env name>
```

### 🔹 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Pipeline (via DVC)

Each pipeline stage is DVC-tracked and reproducible.

| Stage               | Command                                      |
| ------------------- | -------------------------------------------- |
| Data Ingestion      | `python main.py --stage data_ingestion`      |
| Data Validation     | `python main.py --stage data_validation`     |
| Data Cleaning       | `python main.py --stage data_cleaning`       |
| Data Transformation | `python main.py --stage data_transformation` |
| Model Training      | `python main.py --stage model_training`      |
| Model Evaluation    | `python main.py --stage model_evaluation`    |

Run the full pipeline:

```bash
dvc repro
```

---

## 📈 MLflow Tracking (via Dagshub)

- Logs parameters, metrics (R², MAE, MSE), models
- Stores all experiments and best model in the MLflow registry

---

## 🐳 Docker Support

**Build Docker image:**

```bash
docker build -t flight-fare-app .
```

**Run the container:**

```bash
docker run -p 8080:8080 flight-fare-app
```

---

## ⚙️ CI/CD Pipeline (GitHub Actions + AWS ECR)

Your CI/CD workflow includes:

- ✅ Code linting
- ✅ Unit tests (placeholder)
- ✅ Docker image build
- ✅ Image push to AWS ECR
- ✅ Auto-deploy to EC2 container (self-hosted)

Workflow file:

```
.github/workflows/cicd.yml
```

---

## 🧠 Key Highlights

- ✅ End-to-End ML lifecycle pipeline
- ✅ Model tuning via RandomizedSearchCV
- ✅ MLflow-based experiment tracking
- ✅ CI/CD auto-deployment with GitHub → AWS
- ✅ Production-grade FastAPI backend

---

![screencapture-127-0-0-1-8080-2025-08-04-14_05_56](https://github.com/user-attachments/assets/14706d95-7032-4790-8ae2-0f0e169005dc)




## 📄 License

Distributed under the MIT License.\
See `LICENSE` for more information.
