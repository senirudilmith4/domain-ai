import sqlite3
import pandas as pd
from contextlib import contextmanager

class StudentDB:
    # Update the default path to match your file name
    def __init__(self, db_path=r"D:\OneDrive\Documents\IIT\STAGE 02\DSGP\Domain AI\MVP\FrontEnd\students.db"):
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
        """Saves a student's input and ensures all non-taken courses are 0.0, not NULL."""
        
        # 1. Map 'target_year' to 'Stage' to match your Beekeeper table schema
        if 'target_year' in student_dict:
            student_dict['Stage'] = student_dict.pop('target_year')
        
        # 2. Flatten nested user_data dictionary
        if 'user_data' in student_dict:
            user_data = student_dict.pop('user_data')
            student_dict.update(user_data)

        # 3. Remove predicted_gpa to avoid schema mismatch errors
        if 'predicted_gpa' in student_dict:
            del student_dict['predicted_gpa']

        # 4. Define ALL grade columns expected by your database
        # This ensures future courses are recorded as 0.0 even if not in student_dict
        grade_columns = [
            'Grade_CM1601', 'Grade_CM1602', 'Grade_CM1603', 'Grade_CM1604', 'Grade_CM1605', 'Grade_CM1606',
            'Grade_CM2601', 'Grade_CM2602', 'Grade_CM2603', 'Grade_CM2604', 'Grade_CM2605', 'Grade_CM2606', 'Grade_CM2607',
            'Grade_CM3606', 'Grade_CM3604', 'Grade_CM3602', 'Grade_CM3603',
            'Grade_CM4601', 'Grade_CM4605', 'Grade_CM4603', 'Grade_CM4604','Grade_CM4606','Year4_GPA'
        ]

        # 5. THE FIX: Force 0.0 for every possible grade column
        for col in grade_columns:
            # If the column is missing, None, or 'NOT_TAKEN', set it to 0.0
            if col not in student_dict or student_dict[col] is None or student_dict[col] == "NOT_TAKEN":
                student_dict[col] = 0.0
                
        # 6. Convert to DataFrame and save
        df = pd.DataFrame([student_dict])
        
        with self.connection() as conn:
            # 'append' adds the record to your existing Beekeeper table
            df.to_sql('students', conn, if_exists='append', index=False)