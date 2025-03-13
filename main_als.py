from io import StringIO
import logging
import threading
import warnings

from exceptions.BusinessException import BusinessException

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from dotenv import load_dotenv
from utils import als
from utils import shared
import pandas as pd
import pprint

from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()
RECOMMENDATIONS_FILE_NAME = os.getenv('RECOMMENDATIONS_FILE_NAME')

lock = threading.Lock()
recommendation_result_dict = als.load_dict_from_azure(RECOMMENDATIONS_FILE_NAME)
# pprint.pprint(recommendation_result_dict)

@app.errorhandler(Exception)
def handle_exception(ex):
    return shared.handle_exception(ex)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/get_recommendations', methods=['GET'])
@als.require_api_key
def get_recommendations():
    user_id = request.args.get('user_id')

    with lock:
        if user_id is None or recommendation_result_dict.get(user_id) is None:
            return recommendation_result_dict['top_ten_overall_recommendations']

        return recommendation_result_dict[user_id]

@app.route('/train_model', methods=['POST'])
@als.require_api_key
def train_model():
    interactions_file = request.files.get('new_raw_interactions')
    products_file = request.files.get('new_raw_products')

    if not interactions_file:
        raise BusinessException("Interactions file is required")
    if not products_file:
        raise BusinessException("Products file is required")

    new_raw_interactions = pd.read_csv(StringIO(interactions_file.stream.read().decode('utf-8')))
    new_raw_products = pd.read_csv(StringIO(products_file.stream.read().decode('utf-8')))

    new_recommendation_result_dict = als.get_recommendation_result(new_raw_interactions, new_raw_products)

    with lock:
        global recommendation_result_dict
        recommendation_result_dict = new_recommendation_result_dict

    als.save_dictionary_to_azure(RECOMMENDATIONS_FILE_NAME, new_recommendation_result_dict)
    return jsonify({"message": "Model trained and recommendations updated"}), 200

if __name__ == '__main__':
    app.run()