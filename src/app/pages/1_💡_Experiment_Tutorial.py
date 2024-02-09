import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import constants as const
import streamlit_utils as st_utils

st.set_page_config(
    page_title="Tutorial",
    page_icon="💡",
)
#if participant already added feedback to the post questionnaire page, return
if st.session_state.get("feedback_submitted", False):
    switch_page("Post Questionnaire")

# if the participant didnt registered return him to the intro page
if not "participant_registered" in st.session_state or not st.session_state.get("participant_registered", False):
    switch_page("Introduction")

# video_file = open(const.tutorial_video_path, 'rb')  # For a local file
# video_bytes = video_file.read()

st_utils.display_sidebar()
st.markdown(const.tutorial_text, unsafe_allow_html=True)
# st.video(video_bytes)
st.markdown(const.tutorial_finish_text, unsafe_allow_html=True)

if st.button('Go to Experiment'):
    switch_page("Experiment")
    