##🛡️ SecureMail Analyzer - Spam Email Detector

SecureMail Analyzer is a real-time email content analysis tool that detects spam emails using a Machine Learning (ML) model trained on labeled email datasets. The system analyzes message content and predicts whether an email is SPAM or HAM (legitimate).

Built with Python (Scikit-learn, Flask) for the backend and HTML, Tailwind CSS, and JavaScript for the frontend, this project demonstrates both ML implementation and web integration in a simple, educational, and visually interactive way.

#🚀 Features

Machine Learning-Based Detection: Uses trained ML algorithms to classify email content as SPAM or HAM.
High Accuracy: Achieved an accuracy of 96% on the test dataset using a Multinomial Naive Bayes model.
Real-Time Prediction: Instantly analyzes pasted email content and displays classification results.
Dynamic Result Visualization: Displays predictions with color-coded alerts, confidence scores, and icons.
Responsive UI: Built with Tailwind CSS for smooth animations and mobile-friendly layout.
Customizable Model: Supports retraining with new data for improved accuracy.


#🧠 Machine Learning Model

Algorithm Used: Multinomial Naive Bayes
Libraries: Scikit-learn, Pandas, NumPy, Joblib
Dataset: emails.csv (contains labeled email text for spam/ham classification)
Accuracy: ~96% on test data
Training File: train_model.py
Prediction File: predict.py


#Model Workflow

Data Preprocessing: Cleaned and tokenized email text.
Feature Extraction: Used TF-IDF vectorization for text representation.
Model Training: Trained a Multinomial Naive Bayes classifier.
Model Saving: Saved the trained model using Joblib for deployment.
Real-Time Prediction: Integrated with Flask to predict user input in real time.

#💻 Tech Stack
Frontend: HTML, Tailwind CSS, JavaScript
Backend: Python (Flask)
Machine Learning: Scikit-learn, Pandas, NumPy
Icons: Lucide Icons
Model Storage: Joblib

##🧩 Project Structure
Spam_Email_Detector/
│
├── app/
│   ├── app.py                 
│   ├── predict.py             
│   ├── train_model.py        
│   └── __init__.py
│
├── data/
│   └── emails.csv           
│
├── models/
│   └── email_model.joblib    
│
├── template/
│   └── index.html            
│
├── requirements.txt
├── README.md
└── .gitignore


#⚙️ How to Run
Clone the repository:
git clone https://github.com/Shikha18Shukla/Spam_email_detector.git
cd Spam_email_detector


#📊 Example Predictions
Email Example	Prediction	Confidence
"Congratulations! You’ve won a $1000 gift card!"	SPAM	98%
"Reminder: Your meeting is scheduled for tomorrow at 10am."	HAM	94%
"Buy cheap meds online, limited-time offer!"	SPAM	97%

Author: Shikha Shukla
GitHub: Shikha18Shukla


##Frontend :
<img width="1366" height="768" alt="Code_1EHTy1m3g0" src="https://github.com/user-attachments/assets/9b0a75ca-65b5-47bf-985d-b889be893cb1" />
<img width="1366" height="768" alt="brave_jNTQQQk7c3" src="https://github.com/user-attachments/assets/f392fc81-3644-40c3-bf71-bb91dde75bb0" />
<img width="1366" height="768" alt="brave_fNoBW1gX2I" src="https://github.com/user-attachments/assets/6bfdfc38-eb53-422f-b030-7f1671ad9764" />
<img width="1366" height="768" alt="brave_5XnZ2kcKhV" src="https://github.com/user-attachments/assets/bffa3c14-2afa-48f6-9179-4307a85bc05c" />
7_1a5fd76b](https://github.com/user-attachments/assets/2264a5e8-9d03-45cf-9b9a-9a6fd6bbcfbc)



