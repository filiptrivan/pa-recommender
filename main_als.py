import logging
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from dotenv import load_dotenv
from utils.als import *
import pprint

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("implicit")

from functools import wraps
from flask import Flask, request, jsonify


app = Flask(__name__)

load_dotenv()
API_KEY = os.getenv('API_KEY')
ENV = os.getenv('ENV')

recommendation_result_dict = get_recommendation_result()
# pprint.pprint(recommendation_result_dict)

def require_api_key(f):
    @wraps(f) # https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != API_KEY:
            return jsonify({"message": "Unauthorized, invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/GetRecommendations', methods=['GET'])
@require_api_key
def get_recommendations():
    user_id = int(request.args.get('user_id'))

    if user_id is None or recommendation_result_dict.get(user_id) is None:
        return recommendation_result_dict['top_ten_overall_recommendations']

    return recommendation_result_dict[user_id]

app.run()