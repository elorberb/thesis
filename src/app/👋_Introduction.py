import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time
import sqlite3
import config
import re

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

import sqlite3

def save_pre_questionnaire_feedback(feedback):
    # Connect to the SQLite database
    conn = sqlite3.connect(config.EXPERIMENT_DATABASE_FILE)
    cur = conn.cursor()

    # SQL command to insert the feedback data
    sql = '''
    INSERT INTO pre_questionnaire (user_name, email, age, gender)
    VALUES (?, ?, ?, ?)
    '''
    
    # Executing the SQL command with feedback data
    cur.execute(sql, (feedback['user_name'], feedback['email'], feedback['age'], feedback['gender']))
    
    # Committing the changes
    conn.commit()
    
    # Closing the database connection
    conn.close()


markdown_text = f'''
# Trichome Classification Study

This experiment aims to classify cannabis trichomes into clear, cloudy, and amber categories based on your observations. Your input will contribute to the development of a more advanced algorithm, which could be faster and more accurate in predicting the maturity level of cannabis.

## 📋 Pre-Questionnaire
Please answer the following questions before participating:

1. 💼 Full Name
2. 📧 Email
3. 🎂 Age
4. 👫 Gender

## 📝 Instructions
1. ✅ Check the box to indicate your consent to participate in the study.
2. 🚀 Click the "Submit" button to begin.

Your participation is highly valuable, and we appreciate your contribution to this research.

Thank you for your time and involvement!
'''

st.markdown(markdown_text)

if not "switch_to_tutorial_page" in st.session_state:
    st.session_state.switch_to_tutorial_page = False

if st.session_state.switch_to_tutorial_page:
    switch_page("Experiment Tutorial")

with st.form("user_info_form"):
    user_name = st.text_input("Full Name:", placeholder="Please enter your full name")
    email = st.text_input("Email:", placeholder="Please enter your email address")
    age = st.number_input('Age', min_value=18, max_value=100)
    gender = st.selectbox('Gender', ['Female', 'Male', 'Other', 'Prefer not to say'])
    consent = st.checkbox("I agree to participate in the study")

    submitted = st.form_submit_button("Submit")
    if submitted and user_name and consent:
        # Validate the email format using a regular expression
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            st.error("Please enter a valid email address.")
        else:
            st.session_state.user_name = user_name
            st.session_state.email = email
            pre_questionnaire_feedback = {
                'user_name': user_name,
                'email': email, 
                'age': age,
                'gender': gender,
            }
            # save_pre_questionnaire_feedback(pre_questionnaire_feedback)
            st.session_state.switch_to_tutorial_page = True
            st.success("Thank you for participating!")
            time.sleep(2)
            switch_page("Experiment Tutorial")
    elif submitted:
        st.error("Please fill out all fields and agree to participate.")
