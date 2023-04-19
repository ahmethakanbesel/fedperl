import sqlite3
from sqlite3 import Error


def create_connection(db_file, check_same_thread):
    """
        Create a database connection to a SQLite database
        SQLite in Python is multithreading safe.
        It allows multiple threads to access the database simultaneously without causing any conflicts or issues.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file, check_same_thread=check_same_thread)
        conn.execute("PRAGMA foreign_keys = 1")
        return conn
    except Error as e:
        print(e)
    return conn


class Database:

    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = create_connection(self.db_file, False)
        pass

    def get_connection(self):
        return self.conn

    def insert_result(self, model: str, accuracy: float, f1: float, precision: float, recall: float, dataset: str,
                      architecture: str, class_f1_scores: str, class_accuracies: str):
        query = 'INSERT INTO results (model, accuracy, f1, precision, recall, dataset, architecture, class_f1_scores, class_accuracies) VALUES (?, ?, ?, ?, ?, ?, ?,?,?)'
        try:
            # Execute the query, passing in the values
            cursor = self.conn.execute(query, (
            model, accuracy, f1, precision, recall, dataset, architecture, class_f1_scores, class_accuracies))
            # Save the changes to the database
            self.conn.commit()
            # Retrieve the id of the inserted row
            rowid = cursor.lastrowid
            return rowid
        except sqlite3.IntegrityError as error:
            # Handle the UNIQUE constraint error
            print('Error:', error)
            return None
