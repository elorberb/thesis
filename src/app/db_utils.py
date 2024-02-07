import sqlite3
import constants as const
import sqlite3
import constants as const


def create_sqlite_databse():

    print('Creating database...')
    # Connect to SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect(const.EXPERIMENT_DATABASE_FILE)

    print('Database created!')
    # Create cursor object
    cur = conn.cursor()
    print('Creating tables...')
    print('Creating pre_questionnaire table...')
    # Create tables
    cur.execute('''
    CREATE TABLE IF NOT EXISTS pre_questionnaire (
        id INTEGER PRIMARY KEY,
        user_name TEXT,
        email TEXT,
        age INTEGER,
        gender TEXT,
    )
    ''')

    print('Creating experiment_feedback table...')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS experiment_feedback (
        id INTEGER PRIMARY KEY,
        user_name TEXT,
        timestamp TEXT,
        image_path TEXT,
        clear_percentage INTEGER,
        cloudy_percentage INTEGER,
        amber_percentage INTEGER,
        maturity_level INTEGER,
        time_taken_seconds INTEGER
    )
    ''')

    print('Creating post_questionnaire table...')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS post_questionnaire (
        id INTEGER PRIMARY KEY,
        user_name TEXT,
        difficulty TEXT,
        experience_level TEXT,
        cultivation_purpose TEXT,
        vision_quality TEXT,
        ui_feedback TEXT,
        confidence INTEGER,
        educational_background TEXT,
        technical_expertise INTEGER,
        time_of_day TEXT,
        image_impressions TEXT,
        suggestions TEXT,
        assistive_devices BOOLEAN,
        additional_comments TEXT
    )
    ''')

    print('Tables created!')


def save_pre_questionnaire_feedback(feedback):
    conn = sqlite3.connect(const.EXPERIMENT_DATABASE_FILE)
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
    conn = sqlite3.connect(const.EXPERIMENT_DATABASE_FILE)
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
    conn = sqlite3.connect(const.EXPERIMENT_DATABASE_FILE)
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
    