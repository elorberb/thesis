import sqlite3
import config

print('Creating database...')
# Connect to SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(config.EXPERIMENT_DATABASE_FILE)
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
