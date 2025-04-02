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
import math
from exceptions.BusinessException import BusinessException
from utils.classes.Settings import Settings
from utils.classes.StringBuilder import StringBuilder
from utils.emailing import Emailing
import redis

logger = logging.getLogger(__name__)

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

TIMESTAMP_FORMAT = "%d.%m.%Y. %H:%M:%S"

INTERACTION_WEIGHTS = {
    'Bought': 1.0,
    'PutInCart': 0.5,
    'PutInFavorites': 0.3,
    'Clicked': 0.1
}


#region Homepage Recommender Data Manipulation

RECENCY_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.


def get_homepage_interaction_values(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    now = pd.Timestamp.now()
    sb = StringBuilder()
    sb.append(f'Interactions count: {len(raw_interactions)}\n')
    sb.append(f'Raw products not null fields count:\n{raw_products.count()}\n')

    adjust_raw_data(raw_interactions, raw_products)

    raw_interactions['individual_rating'] = get_ratings_column_based_on_recency(now, raw_interactions)

    # FT: Sorted by product_id
    grouped_interactions = raw_interactions.groupby(
        [PRODUCT_COL_NAME, USER_COL_NAME],
        observed=True # FT: Mandatory, if we don't use this, we will get a lot of data from the moment when we casted column to categorical
    ).agg(
        total_rating=('individual_rating', 'sum'),
        interaction_count=('individual_rating', 'count')
    ).reset_index()

    sb.append(f'Grouped interactions of same users and products count: {len(grouped_interactions)}\n')

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
    Emailing().send_email_and_log_info("Homepage recommender data cleaning", sb.__str__())

    return clean_sparse_interactions, user_ids.astype(str), products

# User worked 5 years for one company and was buying only one group of products, now he changed the company and want to buy other group of products, with this function we are forgetting previous interaction
def get_ratings_column_based_on_recency(now: pd.Timestamp, raw_interactions: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME], format=TIMESTAMP_FORMAT)
    diff_days = (now - timestamps) / np.timedelta64(1, 'D')

    if (diff_days < 0).any():
        raise ValueError("The timestamp is in the future. Please provide a valid past timestamp.")

    weights = raw_interactions[INTERACTION_COL_NAME].map(INTERACTION_WEIGHTS)

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

#region Cross Sell Recommender Data Manipulation

PRODUCT_TO_RECOMMEND_COL_NAME = 'product_to_recommend_id'
PRODUCT_FOR_RECOMMENDATION_COL_NAME = 'product_for_recommendation_id'
INTERACTION_WEIGHT_COL_NAME = 'interaction_weight'
SESSION_HOURS = 12
LEFT_SUFFIX = "_l"
RIGHT_SUFFIX = "_r"


def get_cross_sell_interaction_values(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    now = pd.Timestamp.now()
    sb = StringBuilder()
    sb.append(f'Interactions count: {len(raw_interactions)}\n')
    sb.append(f'Raw products not null fields count:\n{raw_products.count()}\n')

    adjust_raw_data(raw_interactions, raw_products)

    raw_interactions[TIMESTAMP_COL_NAME] = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME], format=TIMESTAMP_FORMAT)

    product_product_dataframe = get_product_product_dataframe(raw_interactions)

    if product_product_dataframe.empty:
        raise BusinessException(f'There is no interactions between any of the products within the same user in one {SESSION_HOURS} h session period.')

    product_to_recommend_idx = product_product_dataframe[PRODUCT_TO_RECOMMEND_COL_NAME].cat.codes
    product_for_recommendation_idx = product_product_dataframe[PRODUCT_FOR_RECOMMENDATION_COL_NAME].cat.codes

    clean_sparse_interactions = csr_matrix(
        (product_product_dataframe[INTERACTION_WEIGHT_COL_NAME], (product_to_recommend_idx, product_for_recommendation_idx))
    )

    product_to_recommend_ids = product_product_dataframe[PRODUCT_TO_RECOMMEND_COL_NAME].cat.categories
    product_for_recommendation_ids = product_product_dataframe[PRODUCT_FOR_RECOMMENDATION_COL_NAME].cat.categories

    raw_products_indexed = raw_products.set_index(ID_COL_NAME)

    missing_ids = pd.Index(product_for_recommendation_ids).difference(raw_products_indexed.index)
    if not missing_ids.empty:
        sb.append(f"The product ids that were not found count: {len(missing_ids)}\n")

    products = raw_products_indexed.reindex(product_for_recommendation_ids).reset_index(names=ID_COL_NAME)
    default_values = {'stock': 0, 'status': 'Draft', 'visibility': 'Private', 'active': False }
    products.fillna(value=default_values, inplace=True)

    clean_sparse_interactions = bm25_weight(clean_sparse_interactions, K1=100, B=0.8).tocsr()

    sb.append(get_duration_message(now))
    Emailing().send_email_and_log_info("Cross sell recommender data cleaning", sb.__str__())

    return clean_sparse_interactions, product_to_recommend_ids, products

def get_product_product_dataframe(raw_interactions: pd.DataFrame) -> pd.DataFrame:
    merged_interactions = raw_interactions.merge(
        raw_interactions,
        on=USER_COL_NAME,
        suffixes=(LEFT_SUFFIX, RIGHT_SUFFIX)
    )

    timestamp_left = f"{TIMESTAMP_COL_NAME}{LEFT_SUFFIX}"
    timestamp_right = f"{TIMESTAMP_COL_NAME}{RIGHT_SUFFIX}"
    product_left = f"{PRODUCT_COL_NAME}{LEFT_SUFFIX}"
    product_right = f"{PRODUCT_COL_NAME}{RIGHT_SUFFIX}"

    # Only pairs where the right timestamp is:
    # 1. Later than the left timestamp, OR same timestamp but different product.
    # 2. Within a specified period of the left timestamp.
    condition = (
        (
            (merged_interactions[timestamp_left] < merged_interactions[timestamp_right]) |
            (
                (merged_interactions[timestamp_left] == merged_interactions[timestamp_right]) & 
                (merged_interactions[product_left] != merged_interactions[product_right])
            )
        )
        & (merged_interactions[timestamp_right] < merged_interactions[timestamp_left] + np.timedelta64(SESSION_HOURS, 'h'))
    )

    merged_interactions = merged_interactions[condition]

    merged_interactions['weight'] = get_interaction_weights(merged_interactions)

    product_product_dataframe = merged_interactions\
        .groupby(
            [f'{PRODUCT_COL_NAME}{LEFT_SUFFIX}', f'{PRODUCT_COL_NAME}{RIGHT_SUFFIX}'],
            observed=True # FT: Mandatory, if we don't use this, we will get a lot of data from the moment when we casted column to categorical
        )['weight']\
        .sum()\
        .reset_index()
    
    product_product_dataframe.columns = [PRODUCT_TO_RECOMMEND_COL_NAME, PRODUCT_FOR_RECOMMENDATION_COL_NAME, INTERACTION_WEIGHT_COL_NAME]

    # NOTE FT: We need to remove_unused_categories because we filtered some in previous steps
    product_product_dataframe[PRODUCT_TO_RECOMMEND_COL_NAME] = product_product_dataframe[PRODUCT_TO_RECOMMEND_COL_NAME].cat.remove_unused_categories()
    product_product_dataframe[PRODUCT_FOR_RECOMMENDATION_COL_NAME] = product_product_dataframe[PRODUCT_FOR_RECOMMENDATION_COL_NAME].cat.remove_unused_categories()

    return product_product_dataframe

def get_interaction_weights(merged_interactions: pd.DataFrame) -> np.ndarray:
    seconds_diff = (merged_interactions[f"{TIMESTAMP_COL_NAME}{RIGHT_SUFFIX}"] - merged_interactions[f"{TIMESTAMP_COL_NAME}{LEFT_SUFFIX}"]) / np.timedelta64(1, 's')
    days_diff = seconds_diff / 86400.0

    weights = merged_interactions[f"{INTERACTION_COL_NAME}{RIGHT_SUFFIX}"].map(INTERACTION_WEIGHTS).values

    decayed_weights = weights * np.exp(-((days_diff / RECENCY_DECAY_SCALE) ** 2)) # FT: Faster reduce in first couple of days but as days increase reduce is getting slower and slower
    return decayed_weights

#endregion

#region Shared

def require_api_key(f):
    @wraps(f) # https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != Settings().API_KEY:
            return jsonify({"message": "Unauthorized, invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated

def handle_exception(ex: Exception):
    exception = None
    request_path = request.path if request else None

    if Settings().ENV == 'Dev':
        exception = traceback.format_exc()

    # FT: Ignore this health check path exception https://stackoverflow.com/questions/77921307/getting-a-404-for-get-robots933456-txt
    if Settings().ENV == 'Prod' and request_path == '/robots933456.txt':
        return

    if isinstance(ex, BusinessException):
        code = ex.code
        log_level = logging.WARN
        message = ex.message
    else:
        code = 500
        log_level = logging.ERROR
        message = "An error occurred in the system, our team has been informed and will fix it as soon as possible. Thank you for your patience."
        Emailing().send_email(Settings().EXCEPTION_EMAILS, 'Unhandled exception in pa-recommender', traceback.format_exc())

    logger.log(log_level, f"Request path: '{request_path}'\n{traceback.format_exc()}")

    response = flask.Response(
        response=json.dumps({
            "message": message,
            "exception": exception,
        }),
        status=code,
        mimetype='application/json'
    )

    return response

def adjust_raw_data(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    raw_products[STOCK_COL_NAME] = raw_products[STOCK_COL_NAME].astype(int)
    raw_products[ACTIVE_COL_NAME] = raw_products[ACTIVE_COL_NAME].astype(bool)
    raw_products[PRICE_COL_NAME] = raw_products[PRICE_COL_NAME].astype(float)

    # Pre-cast to category for faster grouping and later extraction of codes
    raw_interactions[PRODUCT_COL_NAME] = raw_interactions[PRODUCT_COL_NAME].astype('category')
    raw_interactions[USER_COL_NAME] = raw_interactions[USER_COL_NAME].astype('category')

def get_duration_message(start_time: pd.Timestamp) -> str:
    return f'Duration: {(pd.Timestamp.now() - start_time).seconds} seconds'

#endregion