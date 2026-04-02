# 📧 🚀 MLOps Spam Detection API (FastAPI + ML)

## 🚀 Project Overview

This project is a Machine Learning based Spam Classifier built with an end-to-end MLOps pipeline.

It classifies text messages as:

* ✅ Spam
* ❌ Not Spam (Ham)

---

## 🛠️ Tech Stack

* Python
* FastAPI
* Scikit-learn
* Pandas
* Uvicorn

---

## ⚙️ Features

* REST API for prediction
* Trained ML model (pickle file)
* Real-time text classification
* Modular code structure

---

## 📂 Project Structure

```
app/
 └── main.py        # FastAPI app
src/
 ├── model.py       # Model logic
 └── train.py       # Training script
models/
 └── spam_model.pkl
data/
 └── spam.csv
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app/main.py
```

---

## 🔥 API Endpoint

POST `/predict`

Example:

```json
{
  "text": "Win a free iPhone now"
}
```

---

## 📊 Output

```json
{
  "label": "spam",
  "score": 0.99
}
```

---

## 🎥 Demo

### Swagger UI

http://127.0.0.1:8000/docs

### Example Prediction

Input:
"Win a free iPhone now"

Output:
Spam (0.99 confidence)


## 🎯 Future Improvements

* Docker deployment
* CI/CD pipeline
* Model monitoring
* Cloud deployment (AWS)

---

## 👨‍💻 Author

Dhivindharshan
