import os
import json
from flask import Flask, render_template, request, jsonify
from .predict import load_model, predict_text

app = Flask(__name__, template_folder='../templates')


try:
    MODEL = load_model()
    print("--- Model loaded successfully. Ready for prediction. ---")
except FileNotFoundError as e:
    MODEL = None
    print(f"ERROR: {e}")
    print("WARNING: Prediction API will not work until the model is trained and saved.")

@app.route("/")
def index():
    """Renders the main beautiful HTML template."""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    API endpoint to receive email content and return spam/ham prediction.
    Expects JSON body: {"text": "The content of the email."}
    """
    if MODEL is None:
        return jsonify({"error": "ML model not available. Please train the model first."}), 503

    try:
        
        data = request.get_json()
        email_text = data.get("text", "")
        
        if not email_text or len(email_text.strip()) < 5:
             return jsonify({"error": "Please provide sufficient email content for analysis."}), 400

    
        label_int = predict_text(MODEL, email_text)
        
        prediction_label = "SPAM" if label_int == 1 else "HAM"
        
        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "text_analyzed_length": len(email_text)
        })

    except Exception as e:
      
        print(f"Prediction error: {e}")
        return jsonify({"error": "An internal server error occurred during prediction."}), 500

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=5000)