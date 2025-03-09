import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import os
sys.path.append(os.path.abspath(".."))  # Adds the project root to sys.path
os.environ['OPENBLAS_NUM_THREADS'] = '1'

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("implicit")

from recommenders.utils.als import *
from flask import Flask, request, jsonify

app = Flask(__name__)

recommendation_result_dict = get_recommendation_result()

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/GetRecommendations')
def get_recommendations():
    user_id = request.args.get('user_id')
    
    if user_id is None or recommendation_result_dict.get(user_id) is None:
        return list(recommendation_result_dict['top_ten_overall_recommendations'])

    return list(recommendation_result_dict[user_id])

if __name__ == '__main__':
    app.run(debug=True)