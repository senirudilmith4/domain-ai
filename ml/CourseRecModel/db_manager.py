import sqlite3
import pandas as pd
from contextlib import contextmanager

class StudentDB:
    # Update the default path to match your file name
    def __init__(self, db_path=r"D:\IIT Stuff\Group Project\DomainSpecifiedAIAssistant project\Project\domain-ai\ml\CourseRecModel\students.db"):
        self.db_path = db_path

    @contextmanager
    def connection(self):
        """Creates a secure connection context."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def load_benchmarks(self, target_year):
        """Pulls GPA data for threshold calculation."""
        query = f"SELECT Year{target_year}_GPA FROM students WHERE Year{target_year}_GPA > 0"
        with self.connection() as conn:
            return pd.read_sql_query(query, conn)

    def save_prediction(self, student_dict):
        """Saves a single student's input and prediction with corrected column names."""
        
        # 1. Map 'target_year' to 'Stage' to match your Beekeeper table
        if 'target_year' in student_dict:
            student_dict['Stage'] = student_dict.pop('target_year')
        
        # 2. If you have nested user_data, flatten it
        if 'user_data' in student_dict:
            user_data = student_dict.pop('user_data')
            student_dict.update(user_data)

        # 3. STRICTLY REMOVE predicted_gpa so it doesn't crash the SQL insert
            if 'predicted_gpa' in student_dict:
                del student_dict['predicted_gpa']

        for key in list(student_dict.keys()):
            if student_dict[key] is None or student_dict[key] == "NOT_TAKEN":
                student_dict[key] = 0.0
                
        # 3. Convert to DataFrame and save
        df = pd.DataFrame([student_dict])
        
        with self.connection() as conn:
            # Now 'Stage' exists in both the DF and the DB
            df.to_sql('students', conn, if_exists='append', index=False)