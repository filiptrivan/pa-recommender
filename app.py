from io import StringIO
import logging
import threading
import warnings
import requests
from exceptions.BusinessException import BusinessException
from utils.classes.Settings import Settings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from utils import als
from utils import shared
from utils import data
import pandas as pd
import numpy as np
import pprint
from azure.monitor.opentelemetry import configure_azure_monitor
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


if Settings().ENV == 'Prod':
    configure_azure_monitor(connection_string=Settings().APPLICATIONINSIGHTS_CONNECTION_STRING)

logging.basicConfig(level=logging.INFO) # FT: Making level for the whole app
logger = logging.getLogger(__name__)

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"]
)

@app.errorhandler(Exception)
def handle_exception(ex):
    return shared.handle_exception(ex)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/get_homepage_recommendations', methods=['GET'])
@shared.require_api_key
def get_homepage_recommendations():
    user_id = request.args.get('user_id')

    if user_id is None or als.HOMEPAGE_RECOMMENDER_REDIS.get(user_id) is None:
        return als.HOMEPAGE_RECOMMENDER_REDIS.get('top_ten_overall_recommendations')

    return als.HOMEPAGE_RECOMMENDER_REDIS.get(user_id)

@app.route('/train_model', methods=['POST'])
@shared.require_api_key
def train_model():
    interactions_file = request.files.get('new_raw_interactions')
    products_file = request.files.get('new_raw_products')

    if not interactions_file:
        raise BusinessException("Interactions file is required")
    if not products_file:
        raise BusinessException("Products file is required")

    new_raw_interactions = pd.read_csv(StringIO(interactions_file.stream.read().decode()))
    new_raw_products = pd.read_csv(StringIO(products_file.stream.read().decode()))

    als.get_homepage_recommendation_result(new_raw_interactions, new_raw_products)
    
    return jsonify({"message": "Model trained and recommendations updated"}), 200

@app.route('/train_model2', methods=['GET'])
@shared.require_api_key
def train_model2():
    interactionsResponse = requests.get('https://localhost:44357/api/PlayertyLoyals/GetInteractions', verify=False)
    productsResponse = requests.get('https://localhost:44357/api/PlayertyLoyals/GetProducts', verify=False)

    if not interactionsResponse:
        raise BusinessException("Interactions file is required")
    if not productsResponse:
        raise BusinessException("Products file is required")

    new_raw_interactions = pd.DataFrame(interactionsResponse.json())
    new_raw_products = pd.DataFrame(productsResponse.json())

    als.get_homepage_recommendation_result(new_raw_interactions, new_raw_products)

    return jsonify({"message": "Model trained and recommendations updated"}), 200

@app.route('/train_model3', methods=['POST'])
@shared.require_api_key
def train_model3():
    interactions_file = request.files.get('new_raw_interactions')
    products_file = request.files.get('new_raw_products')

    if not interactions_file:
        raise BusinessException("Interactions file is required")
    if not products_file:
        raise BusinessException("Products file is required")

    new_raw_interactions = pd.read_csv(StringIO(interactions_file.stream.read().decode()))
    new_raw_products = pd.read_csv(StringIO(products_file.stream.read().decode()))

    als.get_cross_sell_recommendation_result(new_raw_interactions, new_raw_products)

    return jsonify({"message": "Model trained and recommendations updated"}), 200

@app.route('/train_model4', methods=['GET'])
@shared.require_api_key
def train_model4():
    interactionsResponse = requests.get('https://localhost:44357/api/PlayertyLoyals/GetInteractions', verify=False)
    productsResponse = requests.get('https://localhost:44357/api/PlayertyLoyals/GetProducts', verify=False)

    if not interactionsResponse:
        raise BusinessException("Interactions file is required")
    if not productsResponse:
        raise BusinessException("Products file is required")

    new_raw_interactions = pd.DataFrame(interactionsResponse.json())
    new_raw_products = pd.DataFrame(productsResponse.json())

    als.get_cross_sell_recommendation_result(new_raw_interactions, new_raw_products)
    
    return jsonify({"message": "Model trained and recommendations updated"}), 200

if __name__ == '__main__':
    app.run()