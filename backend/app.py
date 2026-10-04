from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE)

# --------------------------------------------------
# Flask setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load saved ML components
# --------------------------------------------------

kmeans = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_model.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

features = joblib.load(
    os.path.join(MODEL_DIR, "features.pkl")
)

# --------------------------------------------------
# Gemini setup
# --------------------------------------------------

gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)
    print("Gemini API key detected")
else:
    gemini_client = None
    print("WARNING: Gemini API key not found")

# Current model used for Gemini text generation
GEMINI_MODEL = "gemini-3.5-flash-lite"

# --------------------------------------------------
# Home
# --------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Credit Card Customer Segmentation API is running"
    })


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "K-Means",
        "clusters": int(kmeans.n_clusters),
        "gemini": gemini_client is not None
    })


# --------------------------------------------------
# K-Means customer segmentation
# --------------------------------------------------

@app.route("/api/segment", methods=["POST"])
def segment_customer():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        # Check required features
        missing_features = [
            feature
            for feature in features
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing": missing_features
            }), 400

        # Maintain exact training feature order
        values = [
            float(data[feature])
            for feature in features
        ]

        X = np.array([values])

        # Standardize
        X_scaled = scaler.transform(X)

        # Predict cluster
        cluster = int(
            kmeans.predict(X_scaled)[0]
        )

        # Segment interpretation
        if cluster == 0:
            segment_name = "High-Activity / High-Balance Customer"
        else:
            segment_name = "Lower-Activity Customer"

        return jsonify({
            "cluster": cluster,
            "segment": segment_name,
            "features": {
                feature: data[feature]
                for feature in features
            }
        })

    except ValueError:
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Gemini AI insights
# --------------------------------------------------

@app.route("/api/ai-insight", methods=["POST"])
def ai_insight():

    try:
        if gemini_client is None:
            return jsonify({
                "error": "Gemini API key is not configured"
            }), 500

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        cluster = data.get("cluster")
        segment = data.get("segment")

        customer_features = data.get("features")

        if cluster is None or segment is None or not customer_features:
            return jsonify({
                "error": "Cluster, segment and features are required"
            }), 400

        prompt = f"""
You are an AI assistant for a Credit Card Customer Segmentation project.

The customer was classified using K-Means clustering.

Cluster: {cluster}
Segment: {segment}

Customer features:
{customer_features}

Important:
- Treat all monetary values as generic currency units.
- Do not use $, €, ₹, or any other currency symbol.
- Do not assume a specific country or currency.
- Only describe patterns supported by the supplied data.

Provide a concise business-oriented analysis.

Include exactly these sections:

1. Customer Profile
2. Key Behavior
3. Recommended Strategy
4. Possible Risk or Attention Area

Explain the segment in simple language.
Do not invent customer information that is not present in the supplied data.
"""

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return jsonify({
            "insight": response.text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Start server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )