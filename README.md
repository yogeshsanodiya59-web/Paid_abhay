# Development of Symptoms Based Disease Prediction System (Flask + Scikit-learn)

A college-level Flask + Scikit-learn web application that predicts diseases from patient symptoms using a trained RandomForest pipeline. Runs entirely on localhost with SQLite.

---

## Quick Start

### 1. Create & activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the ML model

```bash
python ml/train_model.py
```

This will:
- Generate `ml/dataset.csv` (25,000 synthetic records, 30 diseases)
- Train a RandomForest pipeline
- Save `ml/model.pkl`, `ml/vectorizer.pkl`, `ml/encoders.pkl`
- Print training accuracy and classification report

### 4. Run the Flask app

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Folder Structure

```
disease-prediction-app/
├── app.py                  # Flask application (main entry point)
├── database.py             # SQLAlchemy models (User, Prediction)
├── requirements.txt
├── README.md
│
├── ml/
│   ├── train_model.py      # Training script (run this first)
│   ├── preprocess.py       # Shared preprocessing helpers
│   ├── dataset.csv         # Generated after training
│   ├── model.pkl           # Trained pipeline (generated)
│   ├── vectorizer.pkl      # TF-IDF vectorizer (generated)
│   └── encoders.pkl        # LabelEncoder for diseases (generated)
│
├── templates/              # Jinja2 HTML templates
│   ├── base.html
│   ├── home.html
│   ├── register.html
│   ├── login.html
│   ├── predict.html
│   ├── result.html
│   └── dashboard.html
│
├── static/
│   ├── css/style.css       # Custom medical theme
│   └── js/main.js
│
└── instance/
    └── app.db              # SQLite database (auto-created)
```

---

## ML Pipeline

The model is a single `sklearn.pipeline.Pipeline` saved as `ml/model.pkl`:

```
Input Features
    ├── Symptoms (text)  → TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    ├── Gender           → OneHotEncoder(handle_unknown='ignore')
    └── Age, Symptom_Count → passthrough
         ↓
    ColumnTransformer
         ↓
    RandomForestClassifier(n_estimators=200, random_state=42)
         ↓
    Predicted Disease (30 classes) + Confidence %
```

**Key design decision:** The same `ml/preprocess.py` functions (`clean_symptoms`, `symptoms_list_to_string`, `build_feature_dict`) are used at both training time and inference time, ensuring no train-serve skew.

---

## Flask Endpoints

| Route | Method | Auth Required | Description |
|---|---|---|---|
| `/` | GET | No | Home / landing page |
| `/register` | GET, POST | No | User registration |
| `/login` | GET, POST | No | User login |
| `/logout` | GET | Yes | Logout + clear session |
| `/predict` | GET, POST | Yes | Symptom form + prediction |
| `/result` | GET | Yes | Show last prediction result |
| `/dashboard` | GET | Yes | User's prediction history |

---

## Database Schema

**users**
| Column | Type | Notes |
|---|---|---|
| id | INTEGER | Primary key |
| name | TEXT | Full name |
| email | TEXT | Unique |
| password | TEXT | Werkzeug hashed |
| created_at | TIMESTAMP | Auto |

**predictions**
| Column | Type | Notes |
|---|---|---|
| id | INTEGER | Primary key |
| user_id | INTEGER | FK → users.id |
| age | INTEGER | |
| gender | TEXT | |
| symptoms | TEXT | Comma-separated |
| predicted_disease | TEXT | |
| confidence | FLOAT | 0–100 |
| created_at | TIMESTAMP | Auto |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask 3.x, Python 3.9+ |
| ML | Scikit-learn, Pandas, NumPy, Joblib |
| Database | SQLite via SQLAlchemy |
| Auth | Werkzeug (password hashing), Flask sessions |
| Frontend | Bootstrap 5, Bootstrap Icons, Google Fonts (Inter) |

---

## Disclaimer

> This application is built for **educational purposes only** as a college project demo.
> It does **not** constitute medical advice. Always consult a licensed healthcare professional.
