

#with testing


import joblib
import os

MODEL_PATH = os.path.join("models", "email_model.joblib")

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Train first: python -m app.train_model")
    return joblib.load(path)

def predict_text(model, text):
    return model.predict([text])[0]

if __name__ == "__main__":
    model = load_model()

    # Some sample emails to test
    samples = [
        "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!",
        "Reminder: Your meeting is scheduled for tomorrow at 10am.",
        "Lowest price guaranteed! Buy cheap meds online, limited time offer!",
        "Hey, just wanted to check in about our project status."
    ]

    for email in samples:
        label = predict_text(model, email)
        print(f"\nEmail: {email}\nPrediction: {'SPAM' if label == 1 else 'HAM'}")

'''




#without testing 


# app/predict.py
import os
import joblib
import sys
from app.train_model import clean_text  # reuse the same cleaning

MODEL_PATH = os.path.join("models", "email_model.joblib")

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Train first: python app/train_model.py")
    model = joblib.load(path)
    return model

def predict_email(text, model=None):
    if model is None:
        model = load_model()
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    return "spam" if int(pred) == 1 else "ham"

if __name__ == "__main__":
    model = load_model()
    samples = [
        "Congratulations! You have won a free iPhone. Click here to claim.",
        "Hi team, please find attached the minutes of today's meeting."
    ]
    for s in samples:
        print(f"TEXT: {s}")
        print("PRED:", predict_email(s, model))
        print("------")

    # CLI: pass custom message as argument
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
        print("CLI PRED:", predict_email(msg, model))
'''