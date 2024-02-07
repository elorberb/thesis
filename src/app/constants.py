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

This experiment aims to classify cannabis trichomes into clear, cloudy, and amber categories based on your observations, as well as to assess the maturity level of the cannabis flower. Your input is valuable and will contribute to the development of a more advanced algorithm for both classification and maturity assessment. We deeply appreciate your full cooperation and the time you invest in providing detailed feedback. Your insights are crucial to the success of this project and our ability to better understand the maturation process of cannabis flowers.

## 📋 Pre-Questionnaire
Please answer the following questions before participating:

1. 💼 Full Name
2. 📧 Email
3. 🎂 Age
4. 👫 Gender
5. 🏢 Company/Incubator Name
6. 📊 Type of Entity

## 📝 Instructions
1. ✅ Check the box to indicate your consent to participate in the study.
2. 🚀 Click the "Submit" button to begin.

We appreciate your contribution to this research.

Thank you for your time and involvement!
'''

#  ---------- Tutorial page constants ---------- 
image_display_area = utils.get_image_as_base64('pages/images/images_area.png')
maturity_estimate = utils.get_image_as_base64('pages/images/maturity_estimate_slider.png')
sidebar_feedback = utils.get_image_as_base64('pages/images/review_slider.png')
zoom_image = utils.get_image_as_base64('pages/images/zoom_image.png')
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

### 2. User Feedback Sliders Explained

#### Maturity Stage Slider:
- This slider allows you to estimate the maturity stage of the cannabis flower. The scale ranges from "Early Development" to "Over Maturity."
- You can move the slider to the position that best matches your assessment of the flower's maturity stage. In this example, the slider is set to "Nearly Harvest," indicating that the flower is approaching the optimal harvest time.

<div>
    <img src="{maturity_estimate}" alt="Sidebar Feedback" style="width: {width}%; height: auto;">
</div>

#### Trichome Percentage Sliders:
These sliders are used to estimate the proportion of clear, cloudy, and amber trichomes in the image.

Each type of trichome corresponds to a stage of maturity:

- **Clear Trichomes** (%)
- **Cloudy Trichomes** (%)
- **Amber Trichomes** (%)
- To provide your feedback, adjust each slider to reflect the percentage you estimate for each trichome type. The sliders are interactive, and as you move them, the numerical percentage will update accordingly.

<div>
    <img src="{sidebar_feedback}" alt="Sidebar Feedback" style="width: {width}%; height: auto;">
</div>

#### Navigations Buttons
- **➡️ (Next Image) Button**: Use this to send your trichome assessment for the current image and move to the next image.

- **🔎 (Zoom Image) Button**: Click to zoom the image for a closer look, aiding in more precise feedback.

- **🔚 (End Experiment) Button**: Ends the experiment early. You can click this if you need to exit the experiment before reviewing all the images. However, we encourage all participants to complete the assessment of all images provided

## Flow of Reviewing Images

### Step 1: Image Review

- When you enter the experiment page, an image will automatically be loaded for review.
- Observe the trichomes in the image carefully.

### Step 2: Adjusting Sliders

- Based on your observation, use the sliders in the sidebar to estimate the maturity level and the percentages of clear, cloudy, and amber trichomes.

### Step 3: Submitting Your Feedback

- After adjusting the sliders, click on the ➡️ (Next image) button to submit your feedback.
- For your convinience, you can use the 🔎 (Zoom Image) button to zoom the image for a closer look, aiding in more accurate feedback.
- This action will save your inputs and load the next image for review.


### Step 4: Completing the Experiment

- Continue the process of reviewing images and submitting feedback.
- Once all images have been reviewed or you have clicked the 🔚 (End experiment) button, a message will thank you for your participation.


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