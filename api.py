# ============================
#  PHISHNET BACKEND API (FINAL WORKING VERSION)
# ============================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from firebase_admin import credentials, db, initialize_app
from urllib.parse import urlparse
import re
import json, os


# =====================================================
#  FIREBASE SETUP (Render-safe, no local JSON file)
# =====================================================
firebase_key_str = os.environ.get("FIREBASE_KEY")

if not firebase_key_str:
    raise Exception("FIREBASE_KEY not found in environment variables! Add it in Render.")

firebase_key_json = json.loads(firebase_key_str)

cred = credentials.Certificate(firebase_key_json)

initialize_app(cred, {
    'databaseURL': 'https://phishnet-backend-default-rtdb.asia-southeast1.firebasedatabase.app/'
})


# =====================================================
#  FASTAPI SETUP
# =====================================================
app = FastAPI(
    title="PhishNet API",
    description="Detect phishing URLs and messages using ML & rule-based logic",
    version="4.0"
)


# =====================================================
#  LOAD MODELS + VECTORIZERS
# =====================================================
url_model = joblib.load("url_model.pkl")
url_vectorizer = joblib.load("url_vectorizer.pkl")

msg_model = joblib.load("model.pkl")
msg_vectorizer = joblib.load("vectorizer.pkl")


# =====================================================
#  REQUEST MODELS
# =====================================================
class URLInput(BaseModel):
    url: str

class MessageInput(BaseModel):
    message: str

class CombinedInput(BaseModel):
    text: str


# =====================================================
#  RULE-BASED URL CHECK
# =====================================================
def is_dangerous_url(url: str) -> bool:
    """Checks strong phishing indicators only."""
    url_low = url.lower()

    strong_signs = [
        "@",
        ".php?",
        "login-",
        "secure-",
        "-verify",
    ]

    return any(s in url_low for s in strong_signs)


# =====================================================
#  ROOT ENDPOINT
# =====================================================
@app.get("/")
def root():
    return {"message": "Bhaii ! PhishNet API is running successfully !!"}


# =====================================================
#  URL DETECTOR
# =====================================================
@app.post("/detect_url")
def detect_url(data: URLInput):
    url = data.url.strip()
    ref = db.reference("/url_detections")

    # Rule-based first
    if is_dangerous_url(url):
        ref.child(url.replace(".", "_")).set(1)
        return {"url": url, "prediction": 1, "source": "rule_based"}

    # ML model
    features = url_vectorizer.transform([url])
    prediction = int(url_model.predict(features)[0])

    ref.child(url.replace(".", "_")).set(prediction)

    return {"url": url, "prediction": prediction, "source": "model"}


# =====================================================
#  MESSAGE DETECTOR
# =====================================================
@app.post("/detect_message")
def detect_message(data: MessageInput):
    message = data.message.strip()
    ref = db.reference("/message_detections")

    # Firebase cache
    cached = ref.child(message[:50].replace(".", "_")).get()
    if cached is not None:
        return {"message": message, "prediction": cached, "source": "firebase_cache"}

    # ML prediction
    features = msg_vectorizer.transform([message])
    prediction = int(msg_model.predict(features)[0])

    ref.child(message[:50].replace(".", "_")).set(prediction)

    return {"message": message, "prediction": prediction, "source": "model"}


# =====================================================
#  COMBINED DETECTOR
# =====================================================
@app.post("/detect_combined")
def detect_combined(data: CombinedInput):
    text = data.text.strip()
    ref = db.reference("/combined_detections")

    # Extract URL
    urls = re.findall(r"(https?://[^\s]+)", text)
    extracted_url = urls[0] if urls else None

    # URL prediction
    url_prediction = 0
    url_reason = "no_url"

    if extracted_url:
        if is_dangerous_url(extracted_url):
            url_prediction = 1
            url_reason = "rule_based_phishing"
        else:
            url_features = url_vectorizer.transform([extracted_url])
            url_prediction = int(url_model.predict(url_features)[0])
            url_reason = "ml_url_model"

    # Message prediction
    msg_features = msg_vectorizer.transform([text])
    msg_prediction = int(msg_model.predict(msg_features)[0])

    # Final decision
    if extracted_url and url_prediction == 1:
        final_prediction = 1
        source = "url_indicated_phishing"
    elif msg_prediction == 1:
        final_prediction = 1
        source = "message_indicated_spam"
    else:
        final_prediction = 0
        source = "safe"

    # Save in Firebase
    ref.child(text[:50].replace(".", "_")).set(final_prediction)

    return {
        "text": text,
        "extracted_url": extracted_url,
        "url_prediction": url_prediction,
        "url_reason": url_reason,
        "message_prediction": msg_prediction,
        "final_prediction": final_prediction,
        "source": source
    }
