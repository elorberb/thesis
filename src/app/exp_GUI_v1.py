import streamlit as st
import os
import random
import time
from datetime import datetime
import pandas as pd
import config
from pathlib import Path

# Function to save feedback data to a file
def save_image_feedback(data):
    # This is just a placeholder function
    pass

def save_questionnaire_feedback(data):
    # This is just a placeholder function
    pass

def load_random_image():
    # Read the CSV file to get the list of good quality images
    df = pd.read_csv(config.GOOD_QUALITY_IMAGES_CSV)

    # Select a random row from the DataFrame
    random_row = df.sample(n=1).iloc[0]
    image_number = random_row['image_number']
    week, zoom_type = config.find_image_details(image_number)

    # If the week and zoom type are found, construct the image path
    if week and zoom_type:
        image_path = config.get_raw_image_path(week, zoom_type) / f"{image_number}.JPG"
        return str(image_path)
    else:
        return None

def main():
    st.markdown(
            """ 
            <style> 
                /* Style for the sidebar background */
                [data-testid="stSidebar"] {         
                    background-color: #6FAF5F;
                }

                /* Style for the sidebar warning message */
                [data-testid="stSidebar"] .stAlert {
                    background-color: rgba(255, 229, 204, 0.8); /* Light green background */
                    color: #004d00; /* Dark green text */
                    border-left: 5px solid #004d00; /* Dark green left border */
                }
            </style> 
            """, unsafe_allow_html=True)
    with st.sidebar:
        # Ask for the user's name at the beginning
        st.sidebar.header('Enter Your Details')
        user_name = st.sidebar.text_input("Your Name")
        if not user_name:
            st.warning("Please enter your name to proceed.")
            st.stop()

        # Experiment Instructions
        st.sidebar.header('Experiment Instructions')
        st.sidebar.write('''
            Inspect the cannabis flower's image and estimate the percentage of trichomes in each category:

            - **💿Clear Trichomes**: Young, not at peak potency.
            - **⚪ Cloudy Trichomes**: Indicate peak maturity and THC levels.
            - **🟠 Amber Trichomes**: Past peak THC, more sedative effect.

            Adjust the sliders to represent your estimations. Ensure the total percentage equals 100%. Your estimations are crucial for determining the optimal harvest time.
        ''')

        # Informed Consent
        if not st.sidebar.checkbox('I agree to participate in the study'):
            st.sidebar.write('You need to agree to participate to use the app.')
            st.stop()
        
    # Initialize session state for the questionnaire
    if 'show_questionnaire' not in st.session_state:
        st.session_state.show_questionnaire = False

    st.title("Cannabis Trichome Classification Experiment")

    # Start time tracking
    start_time = time.time()
    
    if st.button('Get Image'):
        image_path = load_random_image()
        print(image_path)
        st.image(image_path, caption="Cannabis Image")

        with st.form(key='trichome_feedback'):
            clear_percentage = st.slider('Clear Trichomes (%)', 0, 100, 5)
            cloudy_percentage = st.slider('Cloudy Trichomes (%)', 0, 100, 10)
            amber_percentage = st.slider('Amber Trichomes (%)', 0, 100, 20)
            submit_button = st.form_submit_button(label='Submit Feedback')

            if submit_button:
                # Calculate time taken and format it
                time_taken = round(time.time() - start_time, 2)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Store the feedback
                feedback = {
                    'user_name': user_name,
                    'timestamp': timestamp,
                    'image_path': image_path,
                    'clear_percentage': clear_percentage,
                    'cloudy_percentage': cloudy_percentage,
                    'amber_percentage': amber_percentage,
                    'time_taken_seconds': time_taken,
                }
                
                # Save feedback
                save_image_feedback(feedback)
                
                st.success('Thank you for your feedback!')
                st.write(f"Time taken: {time_taken} seconds")

    if st.button('Finish'):
        # Set the session state to show the questionnaire
        st.session_state.show_questionnaire = True

    if st.session_state.show_questionnaire:
        # Post-task questionnaire
        st.header("Post-Task Questionnaire")

        # Demographic Data Collection
        age = st.number_input('Age', min_value=18, max_value=100)
        gender = st.selectbox('Gender', ['Female', 'Male', 'Other', 'Prefer not to say'])

        difficulty = st.select_slider(
            "How difficult did you find this task?",
            options=["Very Easy", "Easy", "Moderate", "Hard", "Very Hard"]
        )
        enjoyment = st.select_slider(
            "How enjoyable was the task?",
            options=["Not Enjoyable", "Slightly Enjoyable", "Neutral", "Enjoyable", "Very Enjoyable"]
        )
        experience_level = st.selectbox(
            'Experience Level with Cannabis Cultivation and Trichome Inspection',
            ['None', 'Beginner', 'Intermediate', 'Expert']
        )
        cultivation_purpose = st.selectbox(
            'Purpose of Cannabis Cultivation',
            ['Medicinal', 'Recreational', 'Research', 'Other']
        )
        vision_quality = st.selectbox(
            'Do you have any color vision deficiency?',
            ['No', 'Yes', 'Not Sure']
        )
        ui_feedback = st.text_area('Feedback on the User Interface')
        confidence = st.slider(
            'Confidence Level in Your Estimations',
            0, 100, 50
        )
        educational_background = st.text_input('Educational Background')
        technical_expertise = st.slider(
            'Technical Expertise (1: Low - 5: High)',
            1, 5, 3
        )
        time_of_day = st.selectbox(
            'Time of Day When Task Was Performed',
            ['Morning', 'Afternoon', 'Evening', 'Night']
        )
        image_impressions = st.text_area('Subjective Impressions of the Cannabis Images')
        suggestions = st.text_area('Suggestions for Future Research or Improvements')
        assistive_devices = st.checkbox('Did you use any assistive devices (e.g., magnification tools)?')

        # Ensure all fields are filled before proceeding
        if st.button('Submit Feedback'):
            # Collect responses
            feedback = {
                'user_name': user_name,
                'age': age,
                'gender': gender,
                'enjoyment': enjoyment,
                'difficulty': difficulty,
                'experience_level': experience_level,
                'cultivation_purpose': cultivation_purpose,
                'vision_quality': vision_quality,
                'ui_feedback': ui_feedback,
                'confidence': confidence,
                'educational_background': educational_background,
                'technical_expertise': technical_expertise,
                'time_of_day': time_of_day,
                'image_impressions': image_impressions,
                'suggestions': suggestions,
                'assistive_devices': assistive_devices,
                # Add other feedback details as needed
            }
            # Function to save feedback data
            save_questionnaire_feedback(feedback)
            st.success('Thank you for completing the questionnaire!')
        else:
            st.error('Please fill in all the details.')

if __name__ == "__main__":
    main()
    