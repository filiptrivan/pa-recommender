from functools import wraps
import logging
import os
import csv
import json
import traceback
import flask
from scipy.sparse import csr_matrix, coo_matrix
from flask import jsonify, request
import requests
import pandas as pd
from collections import defaultdict
import numpy as np
from math import exp
from math import log1p
from exceptions.BusinessException import BusinessException
from utils.classes.StringBuilder import StringBuilder
from utils.emailing import Emailing
# from implicit.nearest_neighbours import bm25_weight

logger = logging.getLogger(__name__)

#region Data Manipulation

PRODUCT_COL_NAME = 'productId'
USER_COL_NAME = 'userId'
INTERACTION_COL_NAME = 'interaction'
TIMESTAMP_COL_NAME = 'timestamp'

ID_COL_NAME = 'id'
STOCK_COL_NAME = 'stock'
STATUS_COL_NAME = 'status'
VISIBILITY_COL_NAME = 'visibility'
ACTIVE_COL_NAME = 'active'
TITLE_COL_NAME = 'title'
CATEGORIES_COL_NAME = 'categories'
MANUFACTURER_COL_NAME = 'manufacturer'
PRICE_COL_NAME = 'price'

RECENCY_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.
MULTIPLE_INTERACTIONS_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.

def save_interaction_values(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    now = pd.Timestamp.now()
    sb = StringBuilder()
    sb.append(f'Interactions count: {len(raw_interactions)}\n')
    sb.append(f'Raw products not null fields count:\n{raw_products.count()}\n')

    # Pre-cast to category for faster grouping and later extraction of codes
    raw_interactions[PRODUCT_COL_NAME] = raw_interactions[PRODUCT_COL_NAME].astype('category')
    raw_interactions[USER_COL_NAME] = raw_interactions[USER_COL_NAME].astype('category')

    raw_interactions['individual_rating'] = get_ratings_column_based_on_recency(now, raw_interactions)

    grouped_interactions = raw_interactions.groupby(
        [PRODUCT_COL_NAME, USER_COL_NAME],
        observed=True
    ).agg(
        total_rating=('individual_rating', 'sum'),
        interaction_count=('individual_rating', 'count')
    ).reset_index()

    sb.append(f'Grouped interactions of same users and products count: {len(grouped_interactions)}\n')

    bonus = get_multiple_interaction_bonus(grouped_interactions['interaction_count'], grouped_interactions['total_rating'])

    grouped_interactions['total_rating'] += bonus

    product_idx = grouped_interactions[PRODUCT_COL_NAME].cat.codes
    user_idx = grouped_interactions[USER_COL_NAME].cat.codes

    clean_sparse_interactions = csr_matrix(
        (grouped_interactions['total_rating'], (user_idx, product_idx))
    )

    product_ids = grouped_interactions[PRODUCT_COL_NAME].cat.categories
    user_ids = grouped_interactions[USER_COL_NAME].cat.categories
    sb.append(f'Unique product ids in interactions: {len(product_ids)}\n')
    sb.append(f'Unique user ids in interactions: {len(user_ids)}\n')

    raw_products_indexed = raw_products.set_index(ID_COL_NAME)

    missing_ids = pd.Index(product_ids).difference(raw_products_indexed.index)
    if not missing_ids.empty:
        sb.append(f"The product ids that were not found count: {len(missing_ids)}\n")

    products = raw_products_indexed.reindex(product_ids).reset_index(names=ID_COL_NAME)
    default_values = {'stock': 0, 'status': 'Draft', 'visibility': 'Private', 'active': False }
    products.fillna(value=default_values, inplace=True)

    clean_sparse_interactions = bm25_weight(clean_sparse_interactions, K1=100, B=0.8).tocsr()

    sb.append(get_duration_message(now))
    Emailing().send_email_and_log_info("Data cleaning", sb.__str__())

    return clean_sparse_interactions, products, user_ids

# Maybe im not happy with the product that i bought only one time, so for example 2 clicks and 1 put in favorites is stronger than that
# When i buy product a lot of times other products couldn't ever be recommended, so the max for this bonus is 1
# Slower rise and lower initial values
def get_multiple_interaction_bonus(grouped_interactions_count, grouped_interactions_rating):
    return np.where(
        grouped_interactions_count > 1,
        1.0 - (1.0 / (1.0 + np.log1p((grouped_interactions_count - 1.0) * grouped_interactions_rating))),
        0.0
    )

# User worked 5 years for one company and was buying only one group of products, now he changed the company and want to buy other group of products, with this function we are forgetting previous interaction
def get_ratings_column_based_on_recency(now: pd.Timestamp, raw_interactions: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME], format="%d.%m.%Y. %H:%M:%S")
    diff_days = (now - timestamps) / np.timedelta64(1, 'D')

    if (diff_days < 0).any():
        raise ValueError("The timestamp is in the future. Please provide a valid past timestamp.")
    
    interaction_weights = {
        'Bought': 1.0,
        'PutInCart': 0.5,
        'PutInFavorites': 0.3,
        'Clicked': 0.1
    }

    weights = raw_interactions[INTERACTION_COL_NAME].map(interaction_weights)

    if weights.isnull().any():
        raise ValueError("Interaction value doesn't exist (valid: Bought, PutInCart, PutInFavorites, Clicked).")

    decayed_weights = weights * np.exp(-diff_days / RECENCY_DECAY_SCALE) # FT: Faster reduce in first couple of days but as days increase reduce is getting slower and slower
    return decayed_weights

# FT: Copied from the implicit library, only changed so it's working with columns
def bm25_weight(X, K1=100, B=0.8):
    """Weighs each column of a sparse matrix X by BM25 weighting"""
    X = coo_matrix(X)

    N = float(X.shape[1]) # Total number of products
    idf = np.log(N) - np.log1p(np.bincount(X.row)) # Inverse Document Frequency

    # calculate length_norm per column (product)
    col_sums = np.ravel(X.sum(axis=0))
    average_length = col_sums.mean()
    length_norm = (1.0 - B) + B * col_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.col] + X.data) * idf[X.row]
    return X

#endregion

#region Helpers

def require_api_key(f):
    @wraps(f) # https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != os.getenv('API_KEY'):
            return jsonify({"message": "Unauthorized, invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated

def handle_exception(ex: Exception):
    exception = None

    if os.getenv('ENV') == 'Dev':
        exception = traceback.format_exc()

    if isinstance(ex, BusinessException):
        code = ex.code
        log_level = logging.WARN
        message = ex.message
    else:
        code = 500
        log_level = logging.ERROR
        message = "An error occurred in the system, our team has been informed and will fix it as soon as possible. Thank you for your patience."
        Emailing().send_email(os.getenv('EXCEPTION_EMAILS'), 'Unhandled exception in pa-recommender', traceback.format_exc())

    logger.log(log_level, traceback.format_exc())

    response = flask.Response(
        response=json.dumps({
            "message": message,
            "exception": exception,
        }),
        status=code,
        mimetype='application/json'
    )

    return response

def get_duration_message(start_time: pd.Timestamp) -> str:
    return f'Duration: {(pd.Timestamp.now() - start_time).seconds} seconds'

#endregion