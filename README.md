# 📊 Student Result Predictor

Predict student performance using Machine Learning!  

This project demonstrates a complete end-to-end ML workflow: **data → model → web app**.

---

## 🚀 Features
- Predict **Pass / Fail** based on:
  - Study Hours per day  
  - Attendance (%)  
  - Sleep Hours  
- Built using **Python**, **pandas**, **scikit-learn**, and **Streamlit**  
- Interactive **web app** with sliders and real-time prediction

---

## 🛠 Tools & Libraries
- Python 3.x  
- pandas  
- scikit-learn  
- Streamlit  

---

## 💡 How it Works
1. CSV dataset (`student_data.csv`) stores student records  
2. Logistic Regression model trains on the data  
3. Streamlit app lets users input new student info via sliders  
4. Model predicts **Pass** ✅ or **Fail** ❌ instantly  

---

## 📁 Project Structure
```

student-result-predictor/
│
├─ app.py                # Main Streamlit app
├─ student_data.csv      # Dataset
├─ requirements.txt      # Libraries needed
└─ README.md             # Project overview

````

---

## 📈 Demo
Try it live here:  
**[Live App Link](https://student-result-predictor-<your-app-id>.streamlit.app)**

---

## 💼 Resume Line
> Created a Student Result Predictor ML web app using Python, pandas, scikit-learn, and Streamlit, allowing interactive real-time predictions of student performance.

---

## ⚡ How to Run Locally
1. Clone repo:  
```bash
git clone https://github.com/Prekshathoriya/student-result-predictor.git
cd student-result-predictor
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run app:

```bash
streamlit run app.py
```

---

