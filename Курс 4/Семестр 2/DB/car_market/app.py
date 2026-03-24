import redis
from redis.exceptions import RedisError

from datetime import datetime, timezone

from flask import Flask, render_template, redirect, url_for, request, flash
from pymongo import MongoClient

from models import db, Car
from forms import CarForm, CommentForm, SearchForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me-in-production'

# PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'postgresql+psycopg2://car_user:car_pass@localhost:5432/car_market'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# MongoDB
mongo_client = MongoClient(
    'mongodb://localhost:27017/',
    serverSelectionTimeoutMS=2000,
    connectTimeoutMS=2000,
)
mongo_db     = mongo_client['car_market']
comments_col = mongo_db['comments']

# Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

db.init_app(app)

with app.app_context():
    db.create_all()


def avg_rating(product_id):
    cache_key = f"car:{product_id}:avg_rating"

    cached = redis_client.get(cache_key)
    if cached is not None:
        return float(cached)

    try:
        pipeline = [
            {'$match': {'product_id': product_id}},
            {'$group': {'_id': None, 'avg': {'$avg': '$rating'}}}
        ]
        result = list(comments_col.aggregate(pipeline))
        rating = round(result[0]['avg'], 1) if result else 0.0

        redis_client.setex(cache_key, 3600, rating)
        return rating
    except Exception:
        return 0.0

def mongo_available():
    try:
        mongo_client.admin.command('ping')
        return True
    except Exception:
        return False

def safe_get_reviews(product_id):
    try:
        return list(comments_col.find({'product_id': product_id}).sort('created_at', -1))
    except Exception:
        return None

def safe_avg_rating(product_id):
    try:
        return avg_rating(product_id)
    except Exception:
        return None


@app.route('/')
def index():
    cars = Car.query.order_by(Car.id).all()
    return render_template('index.html', cars=cars)


@app.route('/cars/new', methods=['GET', 'POST'])
def car_new():
    form = CarForm()
    if form.validate_on_submit():
        car = Car(
            name        = form.name.data.strip(),
            brand       = form.brand.data.strip(),
            description = form.description.data,
            price       = form.price.data,
            stock       = form.stock.data,
        )
        try:
            db.session.add(car)
            db.session.commit()
            flash('Автомобиль успешно добавлен!', 'success')
            return redirect(url_for('car_detail', product_id=car.id))
        except Exception as e:
            db.session.rollback()
            error_msg = str(e).lower()
            
            if 'uq_cars_brand_name' in error_msg:
                flash('Ошибка: автомобиль с такой маркой и названием уже существует.', 'error')
            elif 'numeric field overflow' in error_msg or 'numeric value out of range' in error_msg:
                flash('Ошибка: слишком большая цена. Введите число меньше 1 000 000 000 ₽.', 'error')
            else:
                flash(f'Ошибка базы данных: {str(e)[:100]}...', 'error')
    return render_template('car_new.html', form=form)

@app.route('/popular')
def popular():
    try:
        popular_ids = redis_client.zrevrange("popular:cars", 0, 9, withscores=True)
        car_ids = [int(car_id) for car_id, _ in popular_ids]
        cars = Car.query.filter(Car.id.in_(car_ids)).all()
        
        cars_dict = {car.id: car for car in cars}
        popular_cars = []
        for car_id, views in popular_ids:
            if int(car_id) in cars_dict:
                popular_cars.append({
                    'car': cars_dict[int(car_id)],
                    'views': int(views)
                })
        
        return render_template('popular.html', popular_cars=popular_cars)
    except RedisError:
        flash('Redis недоступен, популярные авто не загружаются.', 'error')
        return render_template('popular.html', popular_cars=[])



@app.route('/cars/<int:product_id>', methods=['GET', 'POST'])
def car_detail(product_id):
    car = Car.query.get_or_404(product_id)
    
    try:
        views_key = f"car:{product_id}:views"
        redis_client.incr(views_key)
        views = redis_client.get(views_key) or 0
        redis_client.zadd("popular:cars", {str(product_id): views})
    except RedisError:
        pass

    form = CommentForm()

    if form.validate_on_submit():
        try:
            comments_col.insert_one({
                'product_id': product_id,
                'author':     form.author.data.strip() or 'Аноним',
                'text':       form.text.data.strip(),
                'rating':     form.rating.data,
                'created_at': datetime.now(timezone.utc),
            })
            try:
                redis_client.delete(f"car:{product_id}:avg_rating")
            except RedisError:
                pass
            flash('Отзыв добавлен!', 'success')
        except Exception:
            flash('Ошибка: MongoDB недоступна. Отзыв не сохранён.', 'error')
        return redirect(url_for('car_detail', product_id=product_id))

    reviews = safe_get_reviews(product_id)
    rating  = safe_avg_rating(product_id)

    if reviews is None:
        flash('Внимание: MongoDB недоступна, отзывы не отображаются.', 'error')
        reviews = []

    return render_template('car_detail.html', car=car, reviews=reviews,
                           form=form, rating=rating)

@app.route('/search')
def search():
    form    = SearchForm(request.args)
    results = None

    if request.args.get('query') is not None or request.args.get('min_rating'):
        query       = (request.args.get('query') or '').strip()
        brand_query = (request.args.get('brand_query') or '').strip()
        min_rating  = int(request.args.get('min_rating') or 0)

        cars_q = Car.query
        if query:
            cars_q = cars_q.filter(Car.name.ilike(f'%{query}%'))
        if brand_query:
            cars_q = cars_q.filter(Car.brand.ilike(f'%{brand_query}'))

        cars = cars_q.all()

        results = []
        mongo_error = False

        for car in cars:
            r   = safe_avg_rating(car.id)
            avg = r if r is not None else 0
            if r is None:
                mongo_error = True
            if avg >= min_rating:
                results.append({'car': car, 'avg_rating': r})

        if mongo_error:
            flash('Внимание: MongoDB недоступна, фильтрация по рейтингу работает некорректно.', 'error')

    return render_template('search.html', form=form, results=results)

if __name__ == '__main__':
    app.run(debug=True)

