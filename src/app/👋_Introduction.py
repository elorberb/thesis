import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time
import re
import constants as const
import db_utils
import streamlit_utils as st_utils

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)


st.markdown(const.intro_text)


if not "switch_to_tutorial_page" in st.session_state:
    st.session_state.switch_to_tutorial_page = False


if not "user_registered" in st.session_state:
    st.session_state.user_registered = False

if st.session_state.switch_to_tutorial_page:
    switch_page("Experiment Tutorial")
    
st_utils.display_sidebar()

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
            # db_utils.save_pre_questionnaire_feedback(pre_questionnaire_feedback)
            st.session_state.switch_to_tutorial_page = True
            st.session_state.user_registered = True
            st.success("Thank you for participating!")
            time.sleep(2)
            switch_page("Experiment Tutorial")
    elif submitted:
        st.error("Please fill out all fields and agree to participate.")
