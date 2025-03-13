from io import StringIO
import logging
import threading
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

raw_interactions = load_excel_from_azure(os.getenv('INTERACTIONS_CSV_FILE_NAME'))
raw_products = load_excel_from_azure(os.getenv('PRODUCTS_CSV_FILE_NAME'))

lock = threading.Lock()
recommendation_result_dict = get_recommendation_result(raw_interactions, raw_products)
# pprint.pprint(recommendation_result_dict)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/get_recommendations', methods=['GET'])
@require_api_key
def get_recommendations():
    user_id = request.args.get('user_id')

    if user_id is not None:
        user_id = int(user_id)

    with lock:
        if user_id is None or recommendation_result_dict.get(user_id) is None:
            return recommendation_result_dict['top_ten_overall_recommendations']

        return recommendation_result_dict[user_id]

@app.route('/train_model', methods=['GET'])
@require_api_key
def train_model():
    interactions_file = request.files.get('new_raw_interactions')
    products_file = request.files.get('new_raw_products')

    global raw_interactions
    if(interactions_file is not None):
        interactions_data = interactions_file.stream.read().decode('utf-8')
        new_raw_interactions = pd.read_csv(StringIO(interactions_data))
    else:
        new_raw_interactions = raw_interactions

    global raw_products
    if(products_file is not None):
        products_data = products_file.stream.read().decode('utf-8')
        new_raw_products = pd.read_csv(StringIO(products_data))
    else:
        new_raw_products = raw_products

    new_recommendation_result_dict = get_recommendation_result(new_raw_interactions, new_raw_products)

    global recommendation_result_dict
    with lock:
        recommendation_result_dict = new_recommendation_result_dict

    return "Model trained and recommendations updated."

app.run()