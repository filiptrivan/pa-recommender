from functools import wraps
import logging
import os
import csv
import json
import traceback
import flask
from scipy.sparse import csr_matrix
from flask import jsonify, request
import requests
import pandas as pd
from collections import defaultdict
import numpy as np
from math import exp
from math import log1p
from exceptions.BusinessException import BusinessException
from utils.emailing import Emailing

logger = logging.getLogger(__name__)

#region Data Manipulation

PRODUCT_COL_NAME = 'productId'
USER_COL_NAME = 'userId'
INTERACTION_COL_NAME = 'interaction'
TIMESTAMP_COL_NAME = 'timestamp'

ID_COLUMN_NAME = 'id'
STOCK_COLUMN_NAME = 'stock'
STATUS_COLUMN_NAME = 'status'
VISIBILITY_COLUMN_NAME = 'visibility'
ACTIVE_COLUMN_NAME = 'active'
TITLE_COLUMN_NAME = 'title'
CATEGORIES_COLUMN_NAME = 'categories'
MANUFACTURER_COLUMN_NAME = 'manufacturer'
PRICE_COLUMN_NAME = 'price'

RECENCY_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.
MULTIPLE_INTERACTIONS_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.

def save_interaction_values(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    logger.info(f'Interactions count: {len(raw_interactions)}\n')
    logger.info(f'Raw products count: {raw_products.count()}\n')
    now = pd.Timestamp.now()
    logger.info(f'Started training: {now}\n')

    raw_interactions['individual_rating'] = get_ratings_column_based_on_recency(now, raw_interactions)

    aggregated = raw_interactions.groupby(
        [PRODUCT_COL_NAME, USER_COL_NAME]
    ).agg(
        total_rating=('individual_rating', 'sum'),
        interaction_count=('individual_rating', 'count')
    ).reset_index()

    logger.info(f'Grouped interactions of same users and products count: {len(aggregated)}\n')

    aggregated['bonus'] = np.where(
        aggregated['interaction_count'] > 1,
        1 - (1 / (1 + np.log1p((aggregated['interaction_count'] - 1) * aggregated['total_rating']))),
        0
    )

    aggregated['final_rating'] = aggregated['total_rating'] + aggregated['bonus']

    aggregated['product_idx'] = aggregated[PRODUCT_COL_NAME].astype('category').cat.codes
    aggregated['user_idx'] = aggregated[USER_COL_NAME].astype('category').cat.codes

    clean_interactions = csr_matrix(
        (aggregated['final_rating'], (aggregated['product_idx'], aggregated['user_idx']))
    )

    product_ids = aggregated[PRODUCT_COL_NAME].unique()
    user_ids = aggregated[USER_COL_NAME].unique()

    logger.info(f'Unique product ids in interactions: {len(product_ids)}\n')
    logger.info(f'Unique user ids in interactions: {len(user_ids)}\n')

    raw_products.set_index(ID_COLUMN_NAME, inplace=True)

    # FT: Needs to use list for products instead of numpy array because it contains different data types
    products = []
    for product_id in product_ids:
        try:
            product_df = raw_products.loc[[product_id]]
        except:
            product_df = None
        append_product(product_id, product_df, products)

    return clean_interactions, products, user_ids

# Maybe im not happy with the product that i bought only one time, so for example 2 clicks and 1 put in favorites is stronger than that
# When i buy product a lot of times other products couldn't ever be recommended, so the max for this bonus is 1
def get_multiple_interaction_bonus(num_of_interactions, rating):
    if num_of_interactions == 1:
        return 0
    
    # bonus = 1 - (1 / (1 + (num_of_interactions - 1) * rating)) # FT: Slower rise and bigger initial values
    bonus = 1 - (1 / (1 + log1p((num_of_interactions - 1) * rating))) # FT: Slower rise and lower initial values

    return min(bonus, 1)

def get_ratings_column_based_on_recency(now: pd.Timestamp, raw_interactions: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME])
    print(timestamps.dtype)
    timestamps = timestamps.dt.tz_localize(None)

    diff = now - timestamps
    diff_days = diff / np.timedelta64(1, 'D')

    if np.any(diff_days < 0):
        raise ValueError("The timestamp is in the future. Please provide a valid past timestamp.")
    
    interaction_weights = {
        'Bought': 1,
        'PutInCart': 0.5,
        'PutInFavorites': 0.3,
        'Clicked': 0.1
    }

    interactions = raw_interactions[INTERACTION_COL_NAME]

    if not np.all(np.isin(interactions, list(interaction_weights.keys()))):
        raise ValueError("Interaction value doesn't exist (valid: Bought, PutInCart, PutInFavorites, Clicked).")

    vectorized_map = np.vectorize(lambda x: interaction_weights[x])
    weights = vectorized_map(interactions)

    decayed_weights = weights * np.exp(-diff_days / RECENCY_DECAY_SCALE)
    return decayed_weights

# User worked 5 years for one company and was buying only one group of products, now he changed the company and want to buy other group of products, with this function we are forgetting previous interaction
def get_rating_based_on_recency(now: pd.Timestamp, row) -> float:
    # FT: Recency bonus
    timestamp = pd.to_datetime(row[TIMESTAMP_COL_NAME]).tz_localize(None)
    diff_days = (now - timestamp).total_seconds() / (60 * 60 * 24)

    if diff_days < 0:
        raise BusinessException("The timestamp is in the future. Please provide a valid past timestamp.")

    interaction = row[INTERACTION_COL_NAME]
    if interaction == 'Bought':
        return get_recency_bonus(diff_days, 1)
    elif interaction == 'PutInCart':
        return get_recency_bonus(diff_days, 0.5)
    elif interaction == 'PutInFavorites':
        return get_recency_bonus(diff_days, 0.3)
    elif interaction == 'Clicked':
        return get_recency_bonus(diff_days, 0.1)
    else:
        raise BusinessException("Interaction value doesn't exist (valid: Bought, PutInCart, PutInFavorites, Clicked).")

def get_recency_bonus(diff_days, interaction_weight, decay_scale=RECENCY_DECAY_SCALE):
    return interaction_weight * exp(-diff_days / decay_scale) # FT: Faster reduce in first couple of days but as days increase reduce is getting slower and slower

def get_dense_interactions_matrix(clean_interactions: list):
    Y = np.array(clean_interactions)
    
    num_products, num_users = Y.shape

    logger.info("num_products: %d", num_products)
    logger.info("num_users: %d", num_users)

    return Y

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
        Emailing().send_email(os.getenv('EXCEPTION_EMAILS'), traceback.format_exc(), 'Unhandled exception in pa-recommender')

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

def append_product(product_id: str, product_df: pd.DataFrame, products: list):
    if not product_df.empty:
        product = product_df.iloc[0]
        stock = int(product.get(STOCK_COLUMN_NAME, 0))
        status = product.get(STATUS_COLUMN_NAME, 'Draft')
        visibility = product.get(VISIBILITY_COLUMN_NAME, 'Private')
        active = bool(product.get(ACTIVE_COLUMN_NAME, False))
        title = product.get(TITLE_COLUMN_NAME, None)
        categories = product.get(CATEGORIES_COLUMN_NAME, None)
        manufacturer = product.get(MANUFACTURER_COLUMN_NAME, None)
        price = product.get(PRICE_COLUMN_NAME, None)
    else:
        stock = 0
        status = 'Draft'
        visibility = 'Private'
        active = False
        title = None
        categories = None
        manufacturer = None
        price = None

    products.append([product_id, stock, status, visibility, active, title, categories, manufacturer, price])

def try_float(value):
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None

#endregion