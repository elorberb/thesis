import streamlit as st
import pandas as pd
import time
from datetime import datetime
import config
from streamlit_extras.switch_page_button import switch_page
import time
import sqlite3

DEFAULT_MATURITY_LEVEL = 5
DEFAULT_CLEAR_PERCENTAGE = 15
DEFAULT_CLOUDY_PERCENTAGE = 80

st.set_page_config(
    page_title="Experiment",
    page_icon="🔬",
)

# hide streamlit menu button
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def save_experiment_feedback(feedback):
    # Connect to the SQLite database
    conn = sqlite3.connect(config.EXPERIMENT_DATABASE_FILE)
    cur = conn.cursor()

    # SQL command to insert the feedback data
    sql = '''
    INSERT INTO experiment_feedback (user_name, timestamp, image_path, clear_percentage, cloudy_percentage, amber_percentage, maturity_level, time_taken_seconds)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    # Executing the SQL command with feedback data
    cur.execute(sql, (feedback['user_name'], feedback['timestamp'], feedback['image_path'], 
                        feedback['clear_percentage'], feedback['cloudy_percentage'], feedback['amber_percentage'], 
                        feedback['maturity_level'], feedback['time_taken_seconds']))
    
    # Committing the changes
    conn.commit()
    
    # Closing the database connection
    conn.close()

def load_random_image(reviewed_images):
    df = pd.read_csv(config.GOOD_QUALITY_IMAGES_CSV)
    available_images = df[~df['image_number'].isin(reviewed_images)]

    if not available_images.empty:
        random_row = available_images.sample(n=1).iloc[0]
        image_number = random_row['image_number']
        week, zoom_type = config.find_image_details(image_number)

        if week and zoom_type:
            image_path = config.get_raw_image_path(week, zoom_type) / f"{image_number}.JPG"
            return str(image_path), image_number
        else:
            return None, None
    else:
        return None, None

# Initialize the experiment state
if 'started_experiment' not in st.session_state:
    st.session_state.started_experiment = True
    st.session_state.reviewed_images = []
    st.session_state.start_time = time.time()  # Initialize start time here
    st.session_state.current_image_path, st.session_state.current_image_number = load_random_image(st.session_state.reviewed_images)

# Sidebar for feedback
with st.sidebar:
    st.header('User Feedback')
    
    if 'maturity_level' not in st.session_state:
        st.session_state['maturity_level'] = DEFAULT_MATURITY_LEVEL
    if 'clear_percentage' not in st.session_state:
        st.session_state['clear_percentage'] = DEFAULT_CLEAR_PERCENTAGE
    if 'cloudy_percentage' not in st.session_state:
        st.session_state['cloudy_percentage'] = DEFAULT_CLOUDY_PERCENTAGE
    
    maturity_level = st.slider('Maturity Level (0-10)', 0, 10, st.session_state.maturity_level)
    min_val, max_val = st.slider(
        "Set percentages for Clear and Cloudy Trichomes:",
        0, 100, (st.session_state.clear_percentage, st.session_state.cloudy_percentage + st.session_state.clear_percentage)
    )

    # Calculate percentages
    clear_percentage = min_val
    cloudy_percentage = max_val - min_val
    amber_percentage = 100 - clear_percentage - cloudy_percentage

    # Custom HTML for styling the slider with grey, white, and orange colors
    slider_html = f"""
    <div style="width: 100%; height: 20px; background: linear-gradient(to right, grey {min_val}%, white {min_val}% {max_val}%, orange {max_val}%);"></div>
    """
    st.markdown(slider_html, unsafe_allow_html=True)

    # Display percentages
    # st.write(f"Clear: {clear_percentage}%  |  Cloudy: {cloudy_percentage}%   |  Amber: {amber_percentage}%")
    percentages_html = f"""
    <div>
        <span style='color: grey; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold; font-size: 40px;'>{clear_percentage}%</span> | 
        <span style='color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold; font-size: 40px;'>{cloudy_percentage}%</span> | 
        <span style='color: orange; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold; font-size: 40px;'>{amber_percentage}%</span>
    </div>
    """
    st.markdown(percentages_html, unsafe_allow_html=True)



    submit_feedback = st.button('Next Image')

# Process feedback and load next image
if submit_feedback:
    time_taken = round(time.time() - st.session_state.start_time, 2)  # Use start time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reset slider values in session state
    st.session_state.maturity_level = DEFAULT_MATURITY_LEVEL
    st.session_state.clear_percentage = DEFAULT_CLEAR_PERCENTAGE
    st.session_state.cloudy_percentage = DEFAULT_CLOUDY_PERCENTAGE

    feedback = {
        'email': st.session_state.email,
        'timestamp': timestamp,
        'image_path': st.session_state.current_image_path,
        'clear_percentage': clear_percentage,
        'cloudy_percentage': cloudy_percentage,
        'amber_percentage': amber_percentage,
        'maturity_level': maturity_level, 
        'time_taken_seconds': time_taken,
    }
    print(feedback)
    save_experiment_feedback(feedback)
    
    st.session_state.reviewed_images.append(st.session_state.current_image_number)
    print(f"reviewed_images: {st.session_state.reviewed_images}")

    new_image_path, new_image_number = load_random_image(st.session_state.reviewed_images)
    
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
    st.image(st.session_state.current_image_path, caption="Cannabis Image")
    
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''

    st.markdown(hide_img_fs, unsafe_allow_html=True)

else:
    st.write("No more images to review.")
    switch_page("Post Questionnaire")
