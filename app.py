import streamlit as st
import pandas as pd
import joblib
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Marks Prediction Model",
    page_icon="🎓",
    layout="wide"
)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/marks_model.pkl")


model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: yellow;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.result {
    font-size: 32px;
    color: #1e90ff;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🎓 Student Marks Prediction Model</div>', unsafe_allow_html=True)
st.write("An interactive ML-powered web application for predicting students' final marks")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Student Inputs")

study_hours = st.sidebar.slider("📘 Study Hours", 0.0, 16.0, 6.0)
attendance = st.sidebar.slider("📅 Attendance (%)", 0, 100, 75)
assignments = st.sidebar.slider("📝 Assignments Completed", 0, 20, 10)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 0.0, 12.0, 7.0)
past_performance = st.sidebar.slider("📊 Past Performance (%)", 0, 100, 70)

# ---------------- MAIN CONTENT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input Summary")
    st.write(f"• Study Hours: **{study_hours} hrs**")
    st.write(f"• Attendance: **{attendance}%**")
    st.write(f"• Assignments: **{assignments}**")
    st.write(f"• Sleep Hours: **{sleep_hours} hrs**")
    st.write(f"• Past Performance: **{past_performance}%**")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")

    if st.button("Predict Marks"):
        with st.spinner("Analyzing student data..."):
            time.sleep(1.5)

            input_df = pd.DataFrame([{
                "Study_Hours": study_hours,
                "Attendance": attendance,
                "Assignments": assignments,
                "Sleep_Hours": sleep_hours,
                "Past_Performance": past_performance
            }])

            prediction = model.predict(input_df)[0]

            st.markdown(f'<div class="result">{prediction:.2f} Marks</div>', unsafe_allow_html=True)

            if prediction >= 85:
                st.success("🌟 Excellent Performance Expected!")
            elif prediction >= 60:
                st.info("👍 Good Performance Expected")
            else:
                st.warning("⚠️ Needs Improvement")

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------------- METRICS ----------------
m1, m2, m3 = st.columns(3)

m1.metric("📈 Model Type", "Linear Regression")
m2.metric("⚙️ Features Used", "5")
m3.metric("🧠 Deployment", "Streamlit Cloud Ready")

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">Built with ❤️ using Machine Learning & Streamlit</div>', unsafe_allow_html=True)
