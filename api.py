# backend api code
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from firebase_admin import credentials, db, initialize_app
from urllib.parse import urlparse
import re
import json, os

# ============================
#  FIREBASE SETUP
# ============================
with open("firebase_key.json", "r") as f:
    firebase_key = f.read()

cred = credentials.Certificate(json.loads(firebase_key))

initialize_app(cred, {
    'databaseURL': 'https://phishnet-backend-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# ============================
#  FAST API APP
# ============================
app = FastAPI(
    title="PhishNet API",
    description="Detect phishing URLs and messages using ML & Rule-Based logic",
    version="3.0"
)

# ============================
#  LOAD MODELS
# ============================
url_model = joblib.load("url_model.pkl")
url_vectorizer = joblib.load("url_vectorizer.pkl")

msg_model = joblib.load("model.pkl")
msg_vectorizer = joblib.load("vectorizer.pkl")

# ============================
#  DATA MODELS
# ============================
class URLInput(BaseModel):
    url: str

class MessageInput(BaseModel):
    message: str

class CombinedInput(BaseModel):
    text: str


# ============================
#  SAFER RULE-BASED URL CHECK
# ============================
def is_dangerous_url(url: str) -> bool:
    """Only checks VERY strong phishing signs"""
    url_low = url.lower()
    parsed = urlparse(url)

    # Only catch things that are definitely suspicious
    dangerous_patterns = [
        "@",              # email style
        ".php?",          # phishing login pages
        "login-",         # fake login
        "secure-",        # fake secure signals
        "-verify",        # fake verification
    ]

    if any(p in url_low for p in dangerous_patterns):
        return True

    # HTTP is NOT counted as phishing — only a warning (handled by Android)
    return False


# ============================
#  ROOT CHECK
# ============================
@app.get("/")
def root():
    return {"message": "Woah! PhishNet API is running successfully !!"}


# ============================
#  URL DETECTION
# ============================
@app.post("/detect_url")
def detect_url(data: URLInput):
    url = data.url.strip()
    ref = db.reference("/url_detections")

    if is_dangerous_url(url):
        ref.child(url.replace(".", "_")).set(1)
        return {"url": url, "prediction": 1, "source": "rule_based"}

    features = url_vectorizer.transform([url])
    prediction = int(url_model.predict(features)[0])
    ref.child(url.replace(".", "_")).set(prediction)

    return {"url": url, "prediction": prediction, "source": "model"}


# ============================
#  MESSAGE DETECTION
# ============================
@app.post("/detect_message")
def detect_message(data: MessageInput):
    message = data.message.strip()
    ref = db.reference("/message_detections")

    # Cache check
    cached = ref.child(message[:50].replace(".", "_")).get()
    if cached is not None:
        return {"message": message, "prediction": cached, "source": "firebase_cache"}

    features = msg_vectorizer.transform([message])
    prediction = int(msg_model.predict(features)[0])

    ref.child(message[:50].replace(".", "_")).set(prediction)

    return {"message": message, "prediction": prediction, "source": "model"}


# ============================
#  COMBINED DETECTOR **FIXED**
# ============================
@app.post("/detect_combined")
def detect_combined(data: CombinedInput):
    text = data.text.strip()
    ref = db.reference("/combined_detections")

    # 1️⃣ Extract URL from the text
    urls = re.findall(r'(https?://[^\s]+)', text)
    extracted_url = urls[0] if urls else None

    # 2️⃣ URL Prediction
    url_prediction = 0
    url_reason = "no_url_found"

    if extracted_url:
        url_reason = "clean_url"
        
        if is_dangerous_url(extracted_url):
            url_prediction = 1
            url_reason = "dangerous_pattern"
        else:
            # ML URL model
            url_features = url_vectorizer.transform([extracted_url])
            url_prediction = int(url_model.predict(url_features)[0])
            url_reason = "ml_url_model"

    # 3️⃣ Message Prediction
    msg_features = msg_vectorizer.transform([text])
    msg_prediction = int(msg_model.predict(msg_features)[0])

    # 4️⃣ FINAL DECISION (balanced)
    if extracted_url and url_prediction == 1:
        final_prediction = 1
        source = "url_indicated_phishing"
    elif msg_prediction == 1:
        final_prediction = 1
        source = "message_indicated_spam"
    else:
        final_prediction = 0
        source = "safe"

    # 5️⃣ SAVE TO FIREBASE
    ref.child(text[:50].replace(".", "_")).set(final_prediction)

    # 6️⃣ RETURN RESULT
    return {
        "text": text,
        "extracted_url": extracted_url,
        "url_prediction": url_prediction,
        "url_reason": url_reason,
        "message_prediction": msg_prediction,
        "final_prediction": final_prediction,
        "source": source
    }
