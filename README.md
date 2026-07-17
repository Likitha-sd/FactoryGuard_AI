# 🏭 FactoryGuard AI

An end-to-end Industrial Predictive Maintenance System built using Machine Learning, FastAPI, and Streamlit.

FactoryGuard AI predicts potential machine failures from real-time sensor data and automatically performs feature engineering before making predictions. The project demonstrates a production-style ML pipeline with a clean separation between frontend, backend, feature engineering, and model inference.

---

## 📸 Demo

### Dashboard

![Dashboard](assets/dashboard.png)

### Prediction

![Prediction](assets/prediction.png)

### FastAPI Documentation

![Swagger](assets/swagger.png)

---

# 🚀 Features

- Predict machine failures using a trained Random Forest model
- Automatic feature engineering during inference
- FastAPI REST API backend
- Interactive Streamlit dashboard
- Machine-specific statistical normalization
- Production-style project structure
- Modular and reusable codebase

---

# 🏗️ Project Architecture

```
                   Streamlit Dashboard
                           │
                           ▼
                     FastAPI Backend
                           │
                           ▼
                  Request Validation
                           │
                           ▼
             Automatic Feature Engineering
                           │
                           ▼
                  Random Forest Model
                           │
                           ▼
                   Prediction Response
```

---

# 📂 Project Structure

```
FactoryGuard-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Pandas
- NumPy

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- Streamlit

### Development

- Git
- GitHub

---

# 📊 Machine Learning Pipeline

1. Collect machine sensor readings
2. Validate request using FastAPI
3. Generate engineered features automatically
4. Normalize values using machine statistics
5. Load trained Random Forest model
6. Predict machine health
7. Return prediction and failure probability

---

# 📈 Input Features

The application requires only seven raw sensor values.

| Feature | Description |
|----------|-------------|
| Machine ID | Machine identifier |
| Temperature | Temperature (°C) |
| Vibration | Vibration (mm/s) |
| Pressure | Pressure (kPa) |
| RPM | Rotational Speed |
| Power | Power Consumption (kW) |
| Operating Hours | Machine runtime |

All additional engineered features are generated automatically during inference.

---

# 📤 API Endpoint

## POST `/predict`

Example Request

```json
{
    "machine_id": "MACHINE_01",
    "temperature_c": 74.5,
    "vibration_mm_s": 3.2,
    "pressure_kpa": 125.0,
    "rotational_speed_rpm": 1450,
    "power_consumption_kw": 11.5,
    "operating_hours": 2150
}
```

Example Response

```json
{
    "success": true,
    "prediction": "Normal",
    "failure_probability": 0.285
}
```

---

# ▶️ Running the Project

Clone the repository

```bash
git clone https://github.com/Likitha-sd/FactoryGuard-AI.git
```

Move into the project

```bash
cd FactoryGuard-AI
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the API

```bash
uvicorn src.api.app:app --reload
```

Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# 📌 Future Improvements

- SHAP Explainability
- Docker Deployment
- Prediction History
- Unit Testing
- Cloud Deployment
- CI/CD Pipeline

---

# 👩‍💻 Author

**Likitha Sri Maddipatla**

Computer Science Engineering Student

GitHub:
https://github.com/Likitha-sd

---

## ⭐ If you found this project useful, consider giving it a star.