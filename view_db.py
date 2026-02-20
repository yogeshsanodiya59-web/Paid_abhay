import os
import sqlite3

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers, tablefmt=None):
        header_str = " | ".join(headers)
        res = [header_str, "-" * len(header_str)]
        for row in data:
            res.append(" | ".join(str(item) for item in row))
        return "\n".join(res)

def view_database():
    db_path = os.path.join("instance", "app.db")
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # View Users
    print("\n--- USERS TABLE ---")
    cursor.execute("SELECT id, name, email, created_at FROM users")
    users = cursor.fetchall()
    print(tabulate(users, headers=["ID", "Name", "Email", "Created At"]))

    # View Predictions
    print("\n--- PREDICTIONS TABLE ---")
    cursor.execute("SELECT id, user_id, age, gender, predicted_disease, confidence, created_at FROM predictions")
    predictions = cursor.fetchall()
    print(tabulate(predictions, headers=["ID", "User ID", "Age", "Gender", "Predicted Disease", "Confidence (%)", "Created At"]))

    conn.close()

if __name__ == "__main__":
    view_database()
