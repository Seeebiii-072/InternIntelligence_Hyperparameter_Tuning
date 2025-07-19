# predict_student.py

import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Sample input (you can replace these values)
midterm = 0.0
assignment = 30.0
practical = 20.0

# Predict
prediction = model.predict([[midterm, assignment, practical]])

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")
