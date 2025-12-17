import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

# title
st.title("📊 Student Result Predictor")

# load data
data = pd.read_csv("student_data.csv")

X = data[["study_hours", "attendance", "sleep_hours"]]
y = data["result"]

# train model
model = LogisticRegression()
model.fit(X, y)

st.write("Enter student details:")

# user inputs
study_hours = st.slider("Study Hours per day", 0, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.slider("Sleep Hours", 0, 10, 6)

# predict button
if st.button("Predict Result"):
    prediction = model.predict([[study_hours, attendance, sleep_hours]])

    if prediction[0] == 1:
        st.success("✅ Prediction: PASS")
    else:
        st.error("❌ Prediction: FAIL")
