import logging
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from utils.als import *
import pprint

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("implicit")

from flask import Flask, request, jsonify

app = Flask(__name__)

recommendation_result_dict = get_recommendation_result()
# pprint.pprint(recommendation_result_dict)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/GetRecommendations')
def get_recommendations():
    user_id = int(request.args.get('user_id'))

    if user_id is None or recommendation_result_dict.get(user_id) is None:
        return recommendation_result_dict['top_ten_overall_recommendations']

    return recommendation_result_dict[user_id]

if __name__ == '__main__':
    app.run(debug=True)