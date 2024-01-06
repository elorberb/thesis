import streamlit as st
import os
import random

def load_random_image(image_folder):
    images = os.listdir(image_folder)
    random_image = random.choice(images)
    return os.path.join(image_folder, random_image)

def main():
    st.title("Cannabis Trichome Classification Feedback")

    image_folder = '/sise/home/etaylor/images/raw_images/week9_15_06_2023/3x_regular/'
    if st.button('Get Random Image'):
        image_path = load_random_image(image_folder)
        st.image(image_path, caption="Random Cannabis Image")

        with st.form(key='trichome_feedback'):
            clear_percentage = st.slider('Clear Trichomes (%)', 0, 100, 0)
            cloudy_percentage = st.slider('Cloudy Trichomes (%)', 0, 100, 0)
            amber_percentage = st.slider('Amber Trichomes (%)', 0, 100, 0)
            submit_button = st.form_submit_button(label='Submit Feedback')

            if submit_button:
                # Store the feedback
                feedback = {
                    'image_path': image_path,
                    'clear_percentage': clear_percentage,
                    'cloudy_percentage': cloudy_percentage,
                    'amber_percentage': amber_percentage
                }
                # Add code to save feedback to a file or database
                st.success('Thank you for your feedback!')

if __name__ == "__main__":
    main()
