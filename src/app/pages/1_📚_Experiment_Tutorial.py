import streamlit as st
import base64
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(
    page_title="Tutorial",
    page_icon="📚",
)


def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# Replace 'local_image_path' with your local image path
image_display_area = get_image_as_base64('/home/etaylor/code_projects/thesis/src/app/pages/images/images_area.png')
sidebar_feedback = get_image_as_base64('/home/etaylor/code_projects/thesis/src/app/pages/images/review_slider.png')
adjusting_sliders = get_image_as_base64('/home/etaylor/code_projects/thesis/src/app/pages/images/submit_image.png')
submit_feedback = get_image_as_base64('/home/etaylor/code_projects/thesis/src/app/pages/images/finish_review.png')

width = 175

tutorial_content = f"""
# Experiment GUI Tutorial

Welcome to the tutorial for our Trichome Classification Experiment. This guide will help you understand how to interact with the experiment's GUI and efficiently review images.

## Understanding the Experiment Interface

The experiment interface is divided into two main sections:

### 1. Main Area

- This is where the images of cannabis trichomes will be displayed.
- Each image needs to be reviewed and classified.

<div>
    <img src="{image_display_area}" alt="Image Display Area" style="width: {width}%; height: auto;">
</div>

### 2. Sidebar - Feedback and Controls

- Here, you will find sliders to adjust the percentage of different trichome types.
- The sliders include:
    - **Clear Trichomes (%)**: Adjust the percentage of clear trichomes.
    - **Cloudy Trichomes (%)**: Adjust the percentage of cloudy trichomes.
    - **Amber Trichomes (%)**: Adjust the percentage of amber trichomes.
    - **Maturity Level (0-10)**: Indicate the maturity level of the trichomes.

<div>
    <img src="{sidebar_feedback}" alt="Sidebar Feedback" style="width: {width}%; height: auto;">
</div>


## Flow of Reviewing Images

### Step 1: Image Review

- When you enter the experiment page, an image will automatically be loaded for review.
- Observe the trichomes in the image carefully.

### Step 2: Adjusting Sliders

- Based on your observation, use the sliders in the sidebar to estimate the percentages of clear, cloudy, and amber trichomes.
- Make sure the total percentage sums up to 100%.

### Step 3: Submitting Your Feedback

- After adjusting the sliders, click on the **Next Image** button to submit your feedback.
- This action will save your inputs and load the next image for review.

<div>
    <img src="{adjusting_sliders}" alt="Adjusting Sliders" style="width: {width}%; height: auto;">
</div>


### Step 4: Completing the Experiment

- Continue the process of reviewing images and submitting feedback.
- Once all images have been reviewed, a message will thank you for your participation.

<div>
    <img src="{submit_feedback}" alt="Submit Feedback" style="width: {width}%; height: auto;">
</div>
"""

st.markdown(tutorial_content, unsafe_allow_html=True)

# Add the video after the tutorial content
st.write("## Video Demonstration")
st.write("Watch this video to see the process of tagging images in action.")

# Add video from a file or URL
video_file = open('/home/etaylor/code_projects/thesis/src/app/pages/images/exp_tutorial.mp4', 'rb')  # For a local file
video_bytes = video_file.read()
st.video(video_bytes)

st.markdown("""
This tutorial aims to make your experience smooth and straightforward. Your accurate and careful observations are invaluable to our research.

Thank you for participating in our experiment!""", unsafe_allow_html=True)

# Button to navigate to the experiment page
if st.button('Go to Experiment'):
    switch_page("Experiment") 