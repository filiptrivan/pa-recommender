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

@app.route('/train_homepage_and_similar_products_model_by_http_request', methods=['GET'])
@shared.require_api_key
def train_homepage_and_similar_products_model_by_http_request():
    new_raw_interactions = shared.get_interactions_from_external_api()
    new_raw_products = shared.get_products_from_external_api()

    als.process_homepage_and_similar_products_recommendations(new_raw_interactions, new_raw_products)

    return jsonify({"message": "Model trained and recommendations updated"}), 200

@app.route('/train_cross_sell_model_by_http_request', methods=['GET'])
@shared.require_api_key
def train_cross_sell_model_by_http_request():
    new_raw_interactions = shared.get_interactions_from_external_api()
    new_raw_products = shared.get_products_from_external_api()

    als.process_cross_sell_recommendation(new_raw_interactions, new_raw_products)
    
    return jsonify({"message": "Model trained and recommendations updated"}), 200

if __name__ == '__main__':
    app.run()