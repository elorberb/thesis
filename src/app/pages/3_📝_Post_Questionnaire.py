import streamlit as st
import config
import sqlite3

st.set_page_config(
    page_title="Post Questionnaire",
    page_icon="📝",
)

import sqlite3

def save_post_questionnaire_feedback(feedback):
    # Connect to the SQLite database
    conn = sqlite3.connect(config.EXPERIMENT_DATABASE_FILE)  # Replace with your database file path
    cur = conn.cursor()

    # SQL command to insert the feedback data
    sql = '''
    INSERT INTO post_questionnaire (user_name, difficulty, experience_level, cultivation_purpose, vision_quality, ui_feedback, confidence, educational_background, technical_expertise, time_of_day, image_impressions, suggestions, assistive_devices, additional_comments)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    # Executing the SQL command with feedback data
    cur.execute(sql, (feedback['user_name'], feedback['difficulty'], feedback['experience_level'], feedback['cultivation_purpose'], 
                    feedback['vision_quality'], feedback['ui_feedback'], feedback['confidence'], feedback['educational_background'],
                    feedback['technical_expertise'], feedback['time_of_day'], feedback['image_impressions'], feedback['suggestions'],
                    feedback['assistive_devices'], feedback['additional_comments']))
    
    # Committing the changes
    conn.commit()
    
    # Closing the database connection
    conn.close()


# Function to display the questionnaire
def show_questionnaire():
    st.header("Post-Task Questionnaire")

    st.write("""
        Thank you for participating in our experiment. Please take a few moments to provide 
        your feedback. Your responses will help us improve the experiment and understand your experience better.
    """)

    # User Experience and Feedback
    difficulty = st.select_slider("How difficult did you find this task?", ["Very Easy", "Easy", "Moderate", "Hard", "Very Hard"])
    confidence = st.select_slider('Confidence Level in Your Estimations', ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    technical_expertise = st.slider('Technical Expertise (1: Low - 5: High)', 1, 5, 3)
    experience_level = st.selectbox('Experience Level with Cannabis Cultivation and Trichome Inspection', ['None', 'Beginner', 'Intermediate', 'Expert'])
    vision_quality = st.selectbox('Do you have any color vision deficiency?', ['No', 'Yes', 'Not Sure'])
    educational_background = st.text_input('Educational Background')
    suggestions = st.text_area('Suggestions for Future Research or Improvements')
    additional_comments = st.text_area("Any additional comments or feedback")

    if st.button('Submit Feedback'):
        feedback = {
            'email': st.session_state.email,
            'difficulty': difficulty,
            'experience_level': experience_level,
            'vision_quality': vision_quality,
            'confidence': confidence,
            'educational_background': educational_background,
            'technical_expertise': technical_expertise,
            'suggestions': suggestions,
            'additional_comments': additional_comments
        }
        save_post_questionnaire_feedback(feedback)
        st.success('Thank you for completing the questionnaire!')
    else:
        st.error('Please fill in all the details.')

# Call the function to display the questionnaire
show_questionnaire()
