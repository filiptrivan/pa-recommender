from io import StringIO
import pandas as pd
import logging
import warnings
from utils.classes.Settings import Settings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from utils import als
from utils import shared
from utils import data
from azure.monitor.opentelemetry import configure_azure_monitor
from flask import Flask, jsonify
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

@app.route('/test_hyperparameter_optimization', methods=['GET'])
@shared.require_api_key
def test_hyperparameter_optimization():
    """
    Test endpoint for hyperparameter optimization.
    This is for testing purposes only - not used in production.
    """
    # Get sample data for testing
    raw_interactions = pd.read_csv('../data/filtered_interactions.csv')
    raw_products = shared.get_products_from_external_api()
    
    # Get interaction values (this will use the current filtering)
    sparse_user_product_matrix, user_ids, products = shared.get_homepage_and_similar_products_interaction_values(raw_interactions, raw_products)
    
    # Run hyperparameter optimization test
    optimization_results = als.optimize_als_hyperparameters_test(sparse_user_product_matrix)
    
    return jsonify({
        "message": "Hyperparameter optimization test completed",
        "best_params": optimization_results['best_params'],
        "best_score": optimization_results['best_score'],
        "test_info": optimization_results['test_info']
    }), 200

if __name__ == '__main__':
    app.run()