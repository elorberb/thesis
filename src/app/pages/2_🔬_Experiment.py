import streamlit as st
from datetime import datetime
from streamlit_extras.switch_page_button import switch_page
import time
import constants as const
import db_utils
import utils
import streamlit_utils as st_utils


st.set_page_config(
    page_title="Experiment",
    page_icon="🔬",
)

if not "user_registered" in st.session_state or not st.session_state.get("user_registered", False):
    switch_page("Introduction")

st.markdown(const.hide_streamlit_style, unsafe_allow_html=True)

# Initialize the experiment state
if 'started_experiment' not in st.session_state:
    st.session_state.started_experiment = True
    st.session_state.reviewed_images = []
    st.session_state.start_time = time.time()  # Initialize start time here
    st.session_state.current_image_path, st.session_state.current_image_number = utils.load_random_image(st.session_state.reviewed_images)

submit_feedback, clear_percentage, cloudy_percentage, amber_percentage, maturity_level, cultivation_week, finish_experiment = st_utils.display_experiment_sidebar()

# if user wants to finish the experiment earlier
if finish_experiment:
    st.success("Thank you for participating in our experiment.")
    time.sleep(2)
    switch_page("Post Questionnaire")

# Process feedback and load next image
if submit_feedback:
    time_taken = round(time.time() - st.session_state.start_time, 2)  # Use start time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    feedback = {
        'email': st.session_state.email,
        'timestamp': timestamp,
        'image_path': st.session_state.current_image_path,
        'image_number': st.session_state.current_image_number,
        'clear_percentage': clear_percentage,
        'cloudy_percentage': cloudy_percentage,
        'amber_percentage': amber_percentage,
        'maturity_level': maturity_level, 
        'time_taken_seconds': time_taken,
    }
    print(feedback)
    # db_utils.save_experiment_feedback(feedback)
    
    st.session_state.reviewed_images.append(st.session_state.current_image_number)
    print(f"reviewed_images: {st.session_state.reviewed_images}")

    new_image_path, new_image_number = utils.load_random_image(st.session_state.reviewed_images)
    
    if new_image_path:
        st.session_state.current_image_path = new_image_path
        st.session_state.current_image_number = new_image_number
        st.session_state.start_time = time.time()  # Reset start time for the new image
        st.rerun()
    else:
        st.success("You have reviewed all images. Thank you!")
        time.sleep(2)
        st.session_state.current_image_path = None  # Clear the image path

# Display the current image
if st.session_state.current_image_path:
    print(st.session_state.current_image_path)
    st.image(st.session_state.current_image_path, caption="Cannabis Image", output_format="PNG")
    st.markdown(const.hide_img_fs, unsafe_allow_html=True)

else:
    switch_page("Post Questionnaire")
