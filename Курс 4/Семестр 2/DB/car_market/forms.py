from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, DecimalField, IntegerField, SelectField
from wtforms.validators import DataRequired, Optional, NumberRange, Length, Regexp

class CarForm(FlaskForm):
    name        = StringField('Название', validators=[DataRequired(), Length(max=20)])
    brand = StringField('Марка', validators=[
        DataRequired(),
        Length(max=20),
        Regexp(r'^[^\d]+$')
    ])
    description = TextAreaField('Описание', validators=[Optional()])
    price       = DecimalField('Цена',    validators=[DataRequired(), NumberRange(min=0.01)])
    stock       = IntegerField('Количество на складе',
                               validators=[DataRequired(), NumberRange(min=0)])

class CommentForm(FlaskForm):
    author = StringField('Автор (необязательно)', validators=[Optional(), Length(max=50)])
    text   = TextAreaField('Текст отзыва', validators=[DataRequired()])
    rating = SelectField('Оценка', choices=[(1,'1'),(2,'2'),(3,'3'),(4,'4'),(5,'5')],
                         coerce=int, validators=[DataRequired()])

class SearchForm(FlaskForm):
    query       = StringField('Поиск по названию', validators=[Optional()])
    min_rating  = SelectField('Минимальный рейтинг',
                             choices=[(0,'Любой'),(1,'1'),(2,'2'),(3,'3'),(4,'4'),(5,'5')],
                             coerce=int, validators=[Optional()])
    brand_query = StringField('Поиск по марке', validators=[Optional()])
    class Meta:
        csrf = False
