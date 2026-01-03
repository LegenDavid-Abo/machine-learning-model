import joblib
import os

model_path = "lesson_02/model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found. Run train_model.py first.")

model = joblib.load(model_path)

# Example applicant
new_applicant = [[29, 48000, 640]]

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
