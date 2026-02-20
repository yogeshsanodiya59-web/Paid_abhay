"""
database.py
-----------
SQLAlchemy database models and initialization for the Disease Prediction App.
Uses SQLite stored at instance/app.db (Flask convention).
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    """Registered application users."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)  # hashed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship: one user → many predictions
    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def __repr__(self):
        return f"<User {self.email}>"


class Prediction(db.Model):
    """Stores each disease prediction made by a logged-in user."""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    predicted_disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.predicted_disease} ({self.confidence:.1f}%)>"


def init_db(app):
    """
    Initialise the database with the Flask app.
    Creates all tables if they do not already exist.
    """
    db.init_app(app)
    with app.app_context():
        db.create_all()
