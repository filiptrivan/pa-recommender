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
RECOMMENDATIONS_FILE_NAME = os.getenv('RECOMMENDATIONS_FILE_NAME')

lock = threading.Lock()
recommendation_result_dict = load_dict_from_azure(RECOMMENDATIONS_FILE_NAME)
pprint.pprint(recommendation_result_dict)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/get_recommendations', methods=['GET'])
@require_api_key
def get_recommendations():
    user_id = request.args.get('user_id')

    with lock:
        if user_id is None or recommendation_result_dict.get(user_id) is None:
            return recommendation_result_dict['top_ten_overall_recommendations']

        return recommendation_result_dict[user_id]

@app.route('/train_model', methods=['POST'])
@require_api_key
def train_model():
    interactions_file = request.files.get('new_raw_interactions')
    products_file = request.files.get('new_raw_products')

    if(interactions_file is None):
        return "Interactions file can't be empty."
    else:
        interactions_data = interactions_file.stream.read().decode('utf-8')
        new_raw_interactions = pd.read_csv(StringIO(interactions_data))

    if(products_file is None):
        return "Products file can't be empty."
    else:
        products_data = products_file.stream.read().decode('utf-8')
        new_raw_products = pd.read_csv(StringIO(products_data))

    new_recommendation_result_dict = get_recommendation_result(new_raw_interactions, new_raw_products)

    global recommendation_result_dict
    with lock:
        recommendation_result_dict = new_recommendation_result_dict

    save_dictionary_to_azure(RECOMMENDATIONS_FILE_NAME, new_recommendation_result_dict)

    return "Model trained and recommendations updated."

app.run()