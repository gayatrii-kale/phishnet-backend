from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from firebase_admin import credentials, db, initialize_app
from urllib.parse import urlparse

# ---------------- FIREBASE SETUP ----------------
cred = credentials.Certificate("firebase_key.json")  # Your downloaded key file
initialize_app(cred, {
    'databaseURL': 'https://phishnet-backend-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="PhishNet API",
    description="Detect phishing URLs and messages using ML & Rule-Based logic",
    version="2.0"
)

# ---------------- LOAD MODELS ----------------
url_model = joblib.load("url_model.pkl")
url_vectorizer = joblib.load("url_vectorizer.pkl")

msg_model = joblib.load("model.pkl")
msg_vectorizer = joblib.load("vectorizer.pkl")

# ---------------- INPUT SCHEMAS ----------------
class URLInput(BaseModel):
    url: str

class MessageInput(BaseModel):
    message: str

# ---------------- HELPER: RULE-BASED CHECK ----------------
def is_phishy_url(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # ⚠️ Rule 1: Non-HTTPS links are suspicious
    if parsed.scheme == "http":
        return True

    # ⚠️ Rule 2: Suspicious keywords
    suspicious_words = ["reset", "verify", "account", "login", "update",
                        "free", "prize", "gift", "claim", "reward", "win"]
    if any(word in url.lower() for word in suspicious_words):
        return True

    # ⚠️ Rule 3: Too many dots (fake domains like paypal.security.verify.com)
    if domain.count('.') > 2:
        return True

    # ⚠️ Rule 4: Contains @ or weird characters
    if "@" in url or "-" in domain:
        return True

    return False

# ---------------- ROOT ENDPOINT ----------------
@app.get("/")
def root():
    return {"message": "PhishNet API is running successfully 🚀"}

# ---------------- URL DETECTION ----------------
@app.post("/detect_url")
def detect_url(data: URLInput):
    url = data.url.strip()
    ref = db.reference("/url_detections")

    # 🔍 Rule-based phishing detection first
    if is_phishy_url(url):
        ref.child(url.replace(".", "_")).set(1)
        return {"url": url, "prediction": 1, "source": "rule_based"}

    # 🔄 Check cache
    cached = ref.child(url.replace(".", "_")).get()
    if cached is not None:
        return {"url": url, "prediction": cached, "source": "firebase_cache"}

    # 🤖 Predict using ML model
    features = url_vectorizer.transform([url])
    prediction = int(url_model.predict(features)[0])

    # 💾 Save result
    ref.child(url.replace(".", "_")).set(prediction)

    return {"url": url, "prediction": prediction, "source": "model"}

# ---------------- MESSAGE DETECTION ----------------
@app.post("/detect_message")
def detect_message(data: MessageInput):
    message = data.message.strip()
    ref = db.reference("/message_detections")

    # 🔄 Check cache
    cached = ref.child(message[:50].replace(".", "_")).get()
    if cached is not None:
        return {"message": message, "prediction": cached, "source": "firebase_cache"}

    # 🤖 Predict using ML model
    features = msg_vectorizer.transform([message])
    prediction = int(msg_model.predict(features)[0])

    # 💾 Save result
    ref.child(message[:50].replace(".", "_")).set(prediction)

    return {"message": message, "prediction": prediction, "source": "model"}

