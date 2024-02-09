import constants as const
import streamlit as st
import db_utils
from streamlit_extras.switch_page_button import switch_page


def setup_intro_page():

    if not "switch_to_tutorial_page" in st.session_state:
        st.session_state.switch_to_tutorial_page = False

    if not "participant_registered" in st.session_state:
        st.session_state.participant_registered = False

    if st.session_state.switch_to_tutorial_page:
        switch_page("Experiment Tutorial")


def display_sidebar():
    st.markdown(const.sidebar_color_css, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(const.assistance_message_app(const.contact_email), unsafe_allow_html=True)
        

def display_experiment_sidebar():
    st.markdown(const.sidebar_color_css, unsafe_allow_html=True)
    with st.sidebar:
        st.header('Participant Feedback')
        
        if 'maturity_level' not in st.session_state:
            st.session_state['maturity_level'] = const.DEFAULT_MATURITY_LEVEL
        if 'clear_percentage' not in st.session_state:
            st.session_state['clear_percentage'] = const.DEFAULT_CLEAR_PERCENTAGE
        if 'cloudy_percentage' not in st.session_state:
            st.session_state['cloudy_percentage'] = const.DEFAULT_CLOUDY_PERCENTAGE

        maturity_stages = ['Early Development', 'Mid Flower', 'Nearly Harvest', 'At Harvest', 'Over Maturity']
        maturity_level = st.select_slider(
        'Estimate the maturity stage of the flower',
        options=maturity_stages,
        value='Nearly Harvest',
        )
        min_val, max_val = st.slider(
            "Clear % | Cloudy % | Amber % portions:",
            0, 100, (st.session_state.clear_percentage, st.session_state.cloudy_percentage + st.session_state.clear_percentage)
        )

        # Calculate percentages
        clear_percentage = min_val
        cloudy_percentage = max_val - min_val
        amber_percentage = 100 - clear_percentage - cloudy_percentage

        # Custom HTML for styling the slider with grey, white, and orange colors
        st.markdown(const.slider_html(min_val, max_val), unsafe_allow_html=True)
        st.markdown(const.percentages_html(clear_percentage, cloudy_percentage, amber_percentage), unsafe_allow_html=True)
        st.markdown(const.two_buttons_css, unsafe_allow_html=True)
        columns = st.columns(3)
        submit_feedback = columns[0].button('➡️ ️', help="Next image")
        columns[1].link_button('🔎 ', st.session_state.current_image_path, help="Zoom image")
        finish_experiment = columns[2].button('⏸️', help="Pause/End experiment")
        st.markdown(const.assistance_message_app(const.contact_email), unsafe_allow_html=True)
    return submit_feedback, clear_percentage, cloudy_percentage, amber_percentage, maturity_level, finish_experiment

# Function to display the questionnaire
def display_post_questionnaire():
    print(st.session_state.get("feedback_submitted", False))
    if st.session_state.get("feedback_submitted", False):
        # Display thank you message if feedback has already been submitted
        st.header("Thank You!")
        st.write("Thank you for participating in the experiment. Your feedback has been received.")
    else:
        st.header("Post-Task Questionnaire")
        st.write("""
            Thank you for participating in our experiment. Please take a few moments to provide 
            your feedback. Your responses will help us improve the experiment and understand your experience better.
        """)

        # participant Experience and Feedback
        difficulty = st.select_slider("How difficult did you find this task?",
                                      ["Very Easy", "Easy", "Moderate", "Hard", "Very Hard"], value="Moderate")
        confidence = st.select_slider('Confidence level in your estimations',
                                      ['Very Low', 'Low', 'Medium', 'High', 'Very High'], value="Medium")
        experience_level = st.selectbox('Experience level with cannabis cultivation and trichome inspection',
                                        ['None', 'Beginner', 'Intermediate', 'Expert'])
        vision_quality = st.selectbox('Do you have any color vision deficiency?', ['No', 'Yes', 'Not Sure'])

        educational_background = st.selectbox(
            "Select Your Educational Background",
            options=const.education_options
        )
        suggestions = st.text_area('Suggestions for future research or improvements')
        additional_comments = st.text_area("Any additional comments or feedback")

        if st.button('Submit Feedback'):
            post_questionnaire_feedback = {
                'participant_id': st.session_state.participant_id,
                'difficulty': difficulty,
                'experience_level': experience_level,
                'vision_quality': vision_quality,
                'confidence': confidence,
                'educational_background': educational_background,
                'suggestions': suggestions,
                'additional_comments': additional_comments
            }
            db_utils.save_post_questionnaire_feedback(post_questionnaire_feedback)
            st.session_state.feedback_submitted = True  # Set the flag indicating feedback was submitted
            st.success('Thank you for completing the questionnaire!')
