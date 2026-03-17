from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import CheckConstraint

db = SQLAlchemy()

class Car(db.Model):
    __tablename__ = 'cars'

    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(20), nullable=False)
    brand       = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text)
    price       = db.Column(db.Numeric(12, 2), nullable=False)
    stock       = db.Column(db.Integer, nullable=False, default=0)

    __table_args__ = (
        CheckConstraint('price > 0', name='price_positive'),
        CheckConstraint('stock >= 0', name='stock_nonnegative'),
    )

