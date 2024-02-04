import utils

#  ---------- General constants ---------- 
contact_email = "etaylor@post.bgu.ac.il"
tutorial_video_path = "pages/images/exp_tutorial.mp4"
BUCKET_NAME = "trichome_classification_study_storage"
EXPERIMENT_DATABASE_FILE = 'src/app/experiment_db.db'

#  ---------- Functions for messages ---------- 
def assistance_message_app(contact_email: str) -> str:
    return f"For any issue, <a href='mailto:{contact_email}' style='color: black;'>contact us</a> "

#  ---------- CSS constants ---------- 
sidebar_color_css = """  
<style>  
    /* Styling for the entire sidebar */  
    [data-testid=stSidebar] {           
        background-color: #D0F0C0; /* Updated background color */ 
    }  
</style>
"""

two_buttons_css = """
<style>  
    div[data-testid="column"] {  
        width: fit-content !important;  
        flex: unset;  
    }  
    div[data-testid="column"] * {  
        width: fit-content !important;  
    }  
</style>  
"""

#  ---------- Intro page constants ---------- 
intro_text = '''
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

#  ---------- Tutorial page constants ---------- 
image_display_area = utils.get_image_as_base64('pages/images/images_area.png')
sidebar_feedback = utils.get_image_as_base64('pages/images/review_slider.png')
adjusting_sliders = utils.get_image_as_base64('pages/images/submit_image.png')
submit_feedback = utils.get_image_as_base64('pages/images/finish_review.png')

width = 175

tutorial_text = f"""
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

## Video Demonstration
Watch this video to see the process of tagging images in action.
"""

tutorial_finish_text = """
This tutorial aims to make your experience smooth and straightforward. Your accurate and careful observations are invaluable to our research.

Thank you for participating in our experiment!"""

# ---------- Experiment page constants ---------- 

# Default values for the trichomes sliders
DEFAULT_MATURITY_LEVEL = 5
DEFAULT_CLEAR_PERCENTAGE = 15
DEFAULT_CLOUDY_PERCENTAGE = 80

# hide streamlit menu button
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

# hide fullscreen button
hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''
# Custom HTML for styling the slider with grey, white, and orange colors
def percentages_html(clear_percentage: int, cloudy_percentage: int, amber_percentage: int) -> str:
    return f"""
    <div>
        <span style='color: grey; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold; font-size: 40px;'>{clear_percentage}%</span> | 
        <span style='color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold; font-size: 40px;'>{cloudy_percentage}%</span> | 
        <span style='color: orange; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold; font-size: 40px;'>{amber_percentage}%</span>
    </div>
    """

def slider_html(min_val: int, max_val: int) -> str:
    return f"""
    <div style="width: 100%; height: 20px; background: linear-gradient(to right, grey {min_val}%, white {min_val}% {max_val}%, orange {max_val}%);"></div>
    """

# Define the cultivation week options
cultivation_weeks = ['4-6 weeks', '6-8 weeks', '8-10 weeks', '10 weeks+']

# ---------- Post-questionnaire page constants ----------

education_options = [
'High School',
'Associate Degree',
'Bachelor’s Degree',
'Master’s Degree',
'Doctorate or Higher',
'Other'
]