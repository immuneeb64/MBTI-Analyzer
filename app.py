import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import re
from questions import questions

# Load models
model_text_IE = tf.keras.models.load_model('IE_model.h5')
model_text_NS = tf.keras.models.load_model('NS_model.h5')
model_text_FT = tf.keras.models.load_model('FT_model.h5')
model_text_JP = tf.keras.models.load_model('JP_model.h5')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model_questionnaire = tf.keras.models.load_model('questionnaire_ann_model.h5')

# MBTI descriptions
mbti_desc = {
    "INTJ": "The Architect – Strategic thinkers with a plan.",
    "INTP": "The Logician – Analytical and curious.",
    "ENTJ": "The Commander – Bold and strong-willed leaders.",
    "ENTP": "The Debater – Energetic, enthusiastic innovators.",
    "INFJ": "The Advocate – Quiet and visionary.",
    "INFP": "The Mediator – Kind, poetic, and empathetic.",
    "ENFJ": "The Protagonist – Charismatic leaders who inspire.",
    "ENFP": "The Campaigner – Creative, social free spirits.",
    "ISTJ": "The Logistician – Practical and responsible.",
    "ISFJ": "The Defender – Warm, organized, and protective.",
    "ESTJ": "The Executive – Born organizers and leaders.",
    "ESFJ": "The Consul – Popular and empathetic.",
    "ISTP": "The Virtuoso – Bold and practical experimenters.",
    "ISFP": "The Adventurer – Flexible and artistic.",
    "ESTP": "The Entrepreneur – Energetic and perceptive.",
    "ESFP": "The Entertainer – Fun-loving and outgoing."
}

def clean_text(text):
    text = text.lower()
    return re.sub(r"http\S+|[^a-z\s]", "", text)

def predict_text(text):
    vec = vectorizer.transform([clean_text(text)]).toarray()
    ie = int((model_text_IE.predict(vec) > 0.5).astype(int)[0][0])
    ns = int((model_text_NS.predict(vec) > 0.5).astype(int)[0][0])
    ft = int((model_text_FT.predict(vec) > 0.5).astype(int)[0][0])
    jp = int((model_text_JP.predict(vec) > 0.5).astype(int)[0][0])
    return mbti_from_traits(ie, ns, ft, jp)

def predict_questionnaire(ans):
    inp = np.array(ans).reshape(1, -1)
    preds = (model_questionnaire.predict(inp) > 0.5).astype(int)[0]
    return mbti_from_traits(*preds)

def mbti_from_traits(ie, ns, ft, jp):
    mbti = ("I" if ie == 0 else "E") + ("N" if ns == 0 else "S") + ("F" if ft == 0 else "T") + ("J" if jp == 0 else "P")
    return mbti

# --- Streamlit GUI ---
st.set_page_config(page_title="MBTI Analyzer", layout="centered")
st.title("MBTI Personality Analyzer")
mode = st.radio("Choose Prediction Mode:", ["Text Input", "Answer Questions"])

if mode == "Text Input":
    user_input = st.text_area("Enter a paragraph about yourself:", height=200)
    if st.button("Analyze Personality"):
        if not user_input.strip():
            st.warning("Please enter text first.")
        else:
            mbti = predict_text(user_input)
            st.success(f"Your MBTI Type: **{mbti}**")
            st.info(mbti_desc.get(mbti, "Type description not found."))
elif mode == "Answer Questions":
    with st.form("questionnaire_form"):
        st.markdown("**Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree):**")
        answers = []
        for i, q in enumerate(questions):
            val = st.slider(f"{i+1}. {q}", 1, 5, 3)
            answers.append(val)
        submitted = st.form_submit_button("Analyze Personality")
        if submitted:
            mbti = predict_questionnaire(answers)
            st.success(f"Your MBTI Type: **{mbti}**")
            st.info(mbti_desc.get(mbti, "Type description not found."))
