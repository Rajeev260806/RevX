"""
=============================================================
PHASE 2 — BACKEND
RevX — AI-Powered Product Review Platform
Flask API | MongoDB | FFC + LSTM Sentiment Models
=============================================================
Endpoints:
  POST /api/reviews          → submit a review
  GET  /api/products         → list all products
  GET  /api/products/<id>    → single product + its reviews
  POST /api/products         → create a product
  GET  /api/health           → model load check

Run:
  pip install flask pymongo flask-cors torch
  python app.py
=============================================================
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# Adjust MODEL_DIR to point to your RevX/model folder
# ─────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.parent          # RevX/
MODEL_DIR     = BASE_DIR / "model"
DATASET_DIR   = BASE_DIR / "dataset_helpers"

import sys
sys.path.insert(0, str(BASE_DIR))

from dataset_helpers.dataset_explore import load_vocab, PAD_TOKEN
from dataset_helpers.data_tokenize   import tokenize_and_encode, MAX_LEN


# ─────────────────────────────────────────────────────────────
# SECTION 1 — MODEL DEFINITIONS
# Copied here so Flask does not depend on training files
# ─────────────────────────────────────────────────────────────

class DropoutSentimentClassifier(nn.Module):
    """FFC model — Embedding → Pool → Linear → ReLU → Dropout → Linear"""
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes=2, pad_idx=0, dropout_p=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.hidden     = nn.Linear(embed_dim, hidden_dim)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        hidden   = self.relu(self.hidden(embedded))
        return self.classifier(self.dropout(hidden))


class LSTMClassifier(nn.Module):
    """LSTM model — Embedding → LSTM → Dropout → Linear"""
    def __init__(self, vocab_size, embedding_dim=128,
                 hidden_dim=128, output_dim=2, dropout_rate=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm      = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(p=dropout_rate)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded                = self.embedding(x)
        _, (hidden, _)          = self.lstm(embedded)
        return self.fc(self.dropout(hidden[-1]))


# ─────────────────────────────────────────────────────────────
# SECTION 2 — MODEL LOADING
# ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

word_to_idx, _ = load_vocab()
VOCAB_SIZE      = len(word_to_idx)
PAD_IDX         = word_to_idx[PAD_TOKEN]

# Load FFC model
ffc_model, ffc_config = None, {}
try:
    with open(MODEL_DIR / "model_config.json") as f:
        ffc_config = json.load(f)

    ffc_model = DropoutSentimentClassifier(
        vocab_size  = VOCAB_SIZE,
        embed_dim   = ffc_config["embed_dim"],
        hidden_dim  = ffc_config["hidden_dim"],
        dropout_p   = ffc_config.get("dropout_p", 0.3),
        num_classes = 2,
        pad_idx     = PAD_IDX,
    ).to(device)
    ffc_model.load_state_dict(
        torch.load(MODEL_DIR / "final_model.pth", map_location=device)
    )
    ffc_model.eval()
    print(f"[OK] FFC model loaded — val_acc={ffc_config.get('val_accuracy', '?'):.2%}")
except Exception as e:
    print(f"[WARN] FFC model failed to load: {e}")

# Load LSTM model
lstm_model = None
try:
    with open(MODEL_DIR / "lstm_config.json") as f:
        lstm_config = json.load(f)
    lstm_model = LSTMClassifier(**lstm_config).to(device)
    lstm_model.load_state_dict(
        torch.load(MODEL_DIR / "best_lstm_model.pth", map_location=device)
    )
    lstm_model.eval()
    print(f"[OK] LSTM model loaded — hidden_dim={lstm_config['hidden_dim']}")
except Exception as e:
    print(f"[WARN] LSTM model failed to load: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — INFERENCE
# ─────────────────────────────────────────────────────────────

def run_inference(text, model):
    """
    Runs a single text review through a model.
    Works for both FFC and LSTM since both now output (1, 2) logits.

    Returns:
        label      : "POSITIVE" or "NEGATIVE"
        confidence : float 0–1
        scores     : {"POSITIVE": float, "NEGATIVE": float}
    """
    model.eval()
    with torch.no_grad():
        encoded = tokenize_and_encode(text, word_to_idx, MAX_LEN)
        tensor  = torch.tensor([encoded], dtype=torch.long).to(device)
        logits  = model(tensor)                        # (1, 2)
        probs   = torch.softmax(logits, dim=1)[0]      # (2,)
        pred    = probs.argmax().item()
        conf    = probs[pred].item()

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    return label, conf, {
        "POSITIVE": round(probs[1].item(), 4),
        "NEGATIVE": round(probs[0].item(), 4),
    }


def analyse_review(text):
    """
    Runs text through both models and returns combined result.
    If one model is unavailable, returns the other's result only.

    Returns dict with ffc and lstm sub-results plus an ensemble label.
    """
    result = {"text_length": len(text.split())}

    if ffc_model:
        label, conf, scores = run_inference(text, ffc_model)
        result["ffc"] = {"label": label, "confidence": round(conf, 4), "scores": scores}

    if lstm_model:
        label, conf, scores = run_inference(text, lstm_model)
        result["lstm"] = {"label": label, "confidence": round(conf, 4), "scores": scores}

    # Ensemble: average both models' positive probability
    # If only one model loaded, use that result as ensemble
    if ffc_model and lstm_model:
        avg_pos = (
            result["ffc"]["scores"]["POSITIVE"] +
            result["lstm"]["scores"]["POSITIVE"]
        ) / 2
        result["ensemble"] = {
            "label"     : "POSITIVE" if avg_pos >= 0.5 else "NEGATIVE",
            "confidence": round(max(avg_pos, 1 - avg_pos), 4),
            "scores"    : {
                "POSITIVE": round(avg_pos, 4),
                "NEGATIVE": round(1 - avg_pos, 4),
            }
        }
    elif ffc_model:
        result["ensemble"] = result["ffc"]
    elif lstm_model:
        result["ensemble"] = result["lstm"]
    else:
        result["ensemble"] = {"label": "UNKNOWN", "confidence": 0, "scores": {}}

    return result


# ─────────────────────────────────────────────────────────────
# SECTION 4 — MONGODB CONNECTION
# ─────────────────────────────────────────────────────────────

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = "revx"

try:
    client   = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    db       = client[DB_NAME]
    products = db["products"]
    reviews  = db["reviews"]
    print(f"[OK] MongoDB connected — db: {DB_NAME}")
except ConnectionFailure as e:
    print(f"[ERROR] MongoDB connection failed: {e}")
    print("        Start MongoDB with: mongod --dbpath /data/db")
    db, products, reviews = None, None, None


def serialize(doc):
    """Converts MongoDB ObjectId fields to strings for JSON serialization."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc


def compute_product_stats(product_id):
    """
    Recomputes aggregate stats for a product from all its reviews.
    Called after every new review submission.

    Returns dict with: avg_rating, review_count, sentiment_breakdown,
                       avg_ffc_positive, avg_lstm_positive,
                       avg_ensemble_positive, ai_score
    """
    pipeline = [
        {"$match": {"product_id": product_id}},
        {"$group": {
            "_id"                  : "$product_id",
            "avg_rating"           : {"$avg": "$rating"},
            "review_count"         : {"$sum": 1},
            "pos_count"            : {"$sum": {"$cond": [{"$eq": ["$sentiment.ensemble.label", "POSITIVE"]}, 1, 0]}},
            "neg_count"            : {"$sum": {"$cond": [{"$eq": ["$sentiment.ensemble.label", "NEGATIVE"]}, 1, 0]}},
            "avg_ffc_positive"     : {"$avg": "$sentiment.ffc.scores.POSITIVE"},
            "avg_lstm_positive"    : {"$avg": "$sentiment.lstm.scores.POSITIVE"},
            "avg_ensemble_positive": {"$avg": "$sentiment.ensemble.scores.POSITIVE"},
        }}
    ]
    result = list(reviews.aggregate(pipeline))
    if not result:
        return {}

    r = result[0]
    total = r["review_count"]

    # AI score: weighted average of ensemble sentiment (70%) and star rating (30%)
    # Star rating normalised to 0-1 range (stars are 1-5)
    norm_rating  = (r["avg_rating"] - 1) / 4 if r["avg_rating"] else 0
    ai_score_raw = (r["avg_ensemble_positive"] * 0.7) + (norm_rating * 0.3)
    ai_score     = round(ai_score_raw * 100, 1)

    return {
        "avg_rating"           : round(r["avg_rating"], 2),
        "review_count"         : total,
        "sentiment_breakdown"  : {
            "positive": r["pos_count"],
            "negative": r["neg_count"],
            "positive_pct": round(r["pos_count"] / total * 100, 1) if total else 0,
        },
        "avg_ffc_positive"     : round(r["avg_ffc_positive"] or 0, 4),
        "avg_lstm_positive"    : round(r["avg_lstm_positive"] or 0, 4),
        "avg_ensemble_positive": round(r["avg_ensemble_positive"] or 0, 4),
        "ai_score"             : ai_score,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 5 — FLASK APP AND ROUTES
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # allows React (localhost:3000) to call Flask (localhost:5000)


def db_required(f):
    """Decorator — returns 503 if MongoDB is unavailable."""
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if db is None:
            return jsonify({"error": "Database unavailable"}), 503
        return f(*args, **kwargs)
    return wrapper


# ── Health check ───────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status"    : "ok",
        "ffc_model" : ffc_model is not None,
        "lstm_model": lstm_model is not None,
        "database"  : db is not None,
        "vocab_size": VOCAB_SIZE,
        "device"    : device,
    })


# ── Products ───────────────────────────────────────────────────

@app.route("/api/products", methods=["GET"])
@db_required
def get_products():
    """
    Returns all products sorted by ai_score descending.
    Includes aggregate stats computed from reviews.
    """
    all_products = list(products.find().sort("stats.ai_score", -1))
    return jsonify([serialize(p) for p in all_products])


@app.route("/api/products", methods=["POST"])
@db_required
def create_product():
    """
    Creates a new product.

    Body (JSON):
        name        : str  required
        description : str  required
        category    : str  optional
        image_url   : str  optional
    """
    data = request.get_json()
    if not data or not data.get("name") or not data.get("description"):
        return jsonify({"error": "name and description are required"}), 400

    product = {
        "name"       : data["name"].strip(),
        "description": data["description"].strip(),
        "category"   : data.get("category", "General").strip(),
        "image_url"  : data.get("image_url", ""),
        "created_at" : datetime.now(timezone.utc).isoformat(),
        "stats"      : {
            "avg_rating"   : 0,
            "review_count" : 0,
            "ai_score"     : 0,
            "sentiment_breakdown": {"positive": 0, "negative": 0, "positive_pct": 0}
        }
    }

    result = products.insert_one(product)
    product["_id"] = str(result.inserted_id)
    return jsonify(product), 201


@app.route("/api/products/<product_id>", methods=["GET"])
@db_required
def get_product(product_id):
    """
    Returns a single product with all its reviews.
    Reviews sorted newest first.
    """
    try:
        product = products.find_one({"_id": ObjectId(product_id)})
    except Exception:
        return jsonify({"error": "Invalid product ID"}), 400

    if not product:
        return jsonify({"error": "Product not found"}), 404

    product_reviews = list(
        reviews.find({"product_id": product_id}).sort("created_at", -1)
    )

    return jsonify({
        "product": serialize(product),
        "reviews": [serialize(r) for r in product_reviews],
    })


# ── Reviews ────────────────────────────────────────────────────

@app.route("/api/reviews", methods=["POST"])
@db_required
def submit_review():
    """
    Submits a new review for a product.
    Runs sentiment analysis through both models automatically.
    Updates product aggregate stats after insertion.

    Body (JSON):
        product_id  : str   required — MongoDB ObjectId string
        author      : str   required
        text        : str   required — the review content
        rating      : int   required — 1 to 5
    """
    data = request.get_json()

    # Validation
    required = ["product_id", "author", "text", "rating"]
    missing  = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    text   = data["text"].strip()
    rating = int(data["rating"])

    if len(text) < 10:
        return jsonify({"error": "Review text must be at least 10 characters"}), 400
    if not 1 <= rating <= 5:
        return jsonify({"error": "Rating must be between 1 and 5"}), 400

    # Verify product exists
    try:
        product = products.find_one({"_id": ObjectId(data["product_id"])})
    except Exception:
        return jsonify({"error": "Invalid product ID"}), 400

    if not product:
        return jsonify({"error": "Product not found"}), 404

    # Run both ML models
    sentiment = analyse_review(text)

    review = {
        "product_id": data["product_id"],
        "author"    : data["author"].strip(),
        "text"      : text,
        "rating"    : rating,
        "sentiment" : sentiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = reviews.insert_one(review)
    review["_id"] = str(result.inserted_id)

    # Recompute and update product aggregate stats
    stats = compute_product_stats(data["product_id"])
    products.update_one(
        {"_id": ObjectId(data["product_id"])},
        {"$set": {"stats": stats}}
    )

    return jsonify({
        "review" : review,
        "stats"  : stats,
    }), 201


@app.route("/api/reviews/preview", methods=["POST"])
def preview_sentiment():
    """
    Runs sentiment analysis on text WITHOUT saving to database.
    Used by the React frontend to show live sentiment as user types.

    Body (JSON):
        text : str
    """
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    sentiment = analyse_review(text)
    return jsonify({"sentiment": sentiment})


# ─────────────────────────────────────────────────────────────
# SECTION 6 — SEED DATA (run once to populate DB)
# ─────────────────────────────────────────────────────────────

def seed_products():
    """
    Inserts sample products if the products collection is empty.
    Run once on first startup to have data to work with.
    """
    if products.count_documents({}) > 0:
        return
    sample_products = [
        {"name": "Sony WH-1000XM5", "description": "Industry-leading noise cancelling wireless headphones with 30hr battery.", "category": "Electronics"},
        {"name": "Samsung Galaxy S24", "description": "Flagship Android smartphone with AI-powered camera system.", "category": "Electronics"},
        {"name": "Nike Air Max 270", "description": "Lifestyle running shoe with large Air unit for all-day comfort.", "category": "Footwear"},
        {"name": "Kindle Paperwhite", "description": "The thinnest, lightest Kindle with a flush-front design.", "category": "Books & Reading"},
        {"name": "Instant Pot Duo 7-in-1", "description": "Electric pressure cooker that replaces 7 kitchen appliances.", "category": "Kitchen"},
    ]
    for p in sample_products:
        p["created_at"] = datetime.now(timezone.utc).isoformat()
        p["stats"] = {"avg_rating": 0, "review_count": 0, "ai_score": 0,
                      "sentiment_breakdown": {"positive": 0, "negative": 0, "positive_pct": 0}}
    products.insert_many(sample_products)
    print(f"[OK] Seeded {len(sample_products)} sample products")


if __name__ == "__main__":
    if db is not None:
        seed_products()
    app.run(debug=True, port=5000)