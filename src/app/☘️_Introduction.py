import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time
import re
import constants as const
import db_utils
import streamlit_utils as st_utils
import uuid

st.set_page_config(
    page_title="Introduction",
    page_icon="☘️",
)


st.markdown(const.intro_text)
st_utils.setup_intro_page()
st_utils.display_sidebar()

with st.form("participant_info_form"):
    participant_id = str(uuid.uuid4())
    participant_name = st.text_input("Full Name:", placeholder="Please enter your full name")
    email = st.text_input("Email:", placeholder="Please enter your email address")
    age = st.number_input('Age', min_value=18, max_value=100)
    gender = st.selectbox('Gender', ['Female', 'Male', 'Other', 'Prefer not to say'])
    company = st.text_input("Company/Incubator Name:", placeholder="Enter your company or incubator name")
    entity_type = st.selectbox('Type of Entity', ['Incubator', 'Commercial Grower', 'Research Facility', 'Other'])
    consent = st.checkbox("I agree to participate in the study")

    submitted = st.form_submit_button("Submit")
    if submitted:
        # validate all fields
        if not all([participant_name, email, age, gender, company, entity_type, consent]):
            st.error("Please fill out all fields and agree to participate.")
        # Validate the email format using a regular expression
        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            st.error("Please enter a valid email address.")
        else:
            st.session_state.participant_name = participant_name
            st.session_state.participant_id = participant_id
            participant_info = {
                'participant_id': participant_id,
                'participant_name': participant_name,
                'email': email, 
                'age': age,
                'gender': gender,
                'company': company,
                'entity_type': entity_type,
            }
            db_utils.save_participant_registration(participant_info)
            st.session_state.switch_to_tutorial_page = True
            st.session_state.participant_registered = True
            st.success("Thank you for participating!")
            time.sleep(2)
            switch_page("Experiment Tutorial")
