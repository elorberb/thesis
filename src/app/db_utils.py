import constants as const
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import pprint as pp


@st.cache_resource
def connect_to_database():
    cred = credentials.Certificate(const.FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db

def save_participant_registration(document):
    db = connect_to_database()
    participant_id = document.get('participant_id')
    doc_ref = db.collection(const.FIREBASE_PARTICIPANTS_COLLECTION).document(participant_id)
    doc_ref.set(document)
    pp.pprint(f'Participant = {document} registration saved successfully to {const.FIREBASE_PARTICIPANTS_COLLECTION} collection.')

def save_experiment_feedback(document):
    db = connect_to_database()
    participant_id = document.get('participant_id')
    # Ensure there's a unique identifier for each feedback entry, e.g., feedback ID or using a timestamp
    feedback_id = document.get('feedback_id')  # Assuming each feedback has a unique ID
    doc_ref = db.collection(const.FIREBASE_EXPERIMENT_FEEDBACK_COLLECTION).document(participant_id).collection('Feedback').document(feedback_id)
    doc_ref.set(document)
    pp.pprint(f'Experiment feedback = {document} saved successfully to {const.FIREBASE_EXPERIMENT_FEEDBACK_COLLECTION} collection.')

def save_post_questionnaire_feedback(document):
    db = connect_to_database()
    participant_id = document.get('participant_id')
    # Similar to feedback, assume there's a unique identifier for each questionnaire response
    questionnaire_id = document.get('questionnaire_id')  # Assuming each questionnaire response has a unique ID
    doc_ref = db.collection(const.FIREBASE_POST_QUESTIONNAIRE_FEEDBACK_COLLECTION).document(participant_id).collection('QuestionnaireFeedback').document(questionnaire_id)
    doc_ref.set(document)
    pp.pprint(f'Post-questionnaire feedback = {document} saved successfully to {const.FIREBASE_POST_QUESTIONNAIRE_FEEDBACK_COLLECTION} collection.')