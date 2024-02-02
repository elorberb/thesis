import sqlite3
import config

def save_pre_questionnaire_feedback(feedback):
    conn = sqlite3.connect(config.EXPERIMENT_DATABASE_FILE)
    cur = conn.cursor()

    sql = '''
    INSERT INTO pre_questionnaire (user_name, email, age, gender)
    VALUES (?, ?, ?, ?)
    '''
    
    # Executing the SQL command with feedback data
    cur.execute(sql, (feedback['user_name'], feedback['email'], feedback['age'], feedback['gender']))
    
    conn.commit()
    conn.close()
    
    
def save_experiment_feedback(feedback):
    conn = sqlite3.connect(config.EXPERIMENT_DATABASE_FILE)
    cur = conn.cursor()

    sql = '''
    INSERT INTO experiment_feedback (user_name, timestamp, image_path, clear_percentage, cloudy_percentage, amber_percentage, maturity_level, time_taken_seconds)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    cur.execute(sql, (feedback['user_name'], feedback['timestamp'], feedback['image_path'], 
                        feedback['clear_percentage'], feedback['cloudy_percentage'], feedback['amber_percentage'], 
                        feedback['maturity_level'], feedback['time_taken_seconds']))
    
    conn.commit()
    conn.close()
    
    
def save_post_questionnaire_feedback(feedback):
    conn = sqlite3.connect(config.EXPERIMENT_DATABASE_FILE)
    cur = conn.cursor()

    sql = '''
    INSERT INTO post_questionnaire (user_name, difficulty, experience_level, cultivation_purpose, vision_quality, ui_feedback, confidence, educational_background, technical_expertise, time_of_day, image_impressions, suggestions, assistive_devices, additional_comments)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    cur.execute(sql, (feedback['user_name'], feedback['difficulty'], feedback['experience_level'], feedback['cultivation_purpose'], 
                    feedback['vision_quality'], feedback['ui_feedback'], feedback['confidence'], feedback['educational_background'],
                    feedback['technical_expertise'], feedback['time_of_day'], feedback['image_impressions'], feedback['suggestions'],
                    feedback['assistive_devices'], feedback['additional_comments']))
    
    conn.commit()
    conn.close()
    