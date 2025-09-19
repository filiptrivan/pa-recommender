from functools import wraps
import logging
import json
import traceback
import flask
from scipy.sparse import csr_matrix, coo_matrix
from flask import jsonify, request
import requests
import pandas as pd
import numpy as np
from exceptions.BusinessException import BusinessException
from utils.classes.Settings import Settings
from utils.classes.StringBuilder import StringBuilder
from utils.emailing import Emailing
import io
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils.outlier_detection import outlier_detection, get_outlier_detection_summary
from utils.outlier_config import OutlierDetectionConfig
from implicit.nearest_neighbours import bm25_weight

logger = logging.getLogger(__name__)

PRODUCT_COL_NAME = 'product_id'
USER_COL_NAME = 'user_uid'
INTERACTION_COL_NAME = 'action'
TIMESTAMP_COL_NAME = 'created'

ID_COL_NAME = 'id'
STOCK_COL_NAME = 'stock'
STATUS_COL_NAME = 'status'
TITLE_COL_NAME = 'title'

INTERACTION_WEIGHTS = {
    'purchase': 1.0,
    'initiate_checkout': 0.7,
    'add_to_cart': 0.5,
    'add_to_wishlist': 0.3,
    'content_view': 0.1
}

# Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.
# By not weighting again metrics improved by 100%, we don't need to weight again if we already made different values for INTERACTION_WEIGHTS
INTERACTION_DECAY_SCALE = 25

# Interaction decay parameters for repeated user-product interactions
# Controls how much each subsequent interaction with the same product is weighted down
INTERACTION_REPEAT_DECAY_FACTOR = 0.8  # Each repeat interaction gets 80% of previous weight

EXTERNAL_API_HEADERS = {
    "Authorization": f"Bearer {Settings().BEARER_TOKEN}"
}
EXTERNAL_API_NAMESPACE='prodavnicaalata'

#region Homepage And Similar Products Recommenders Data Manipulation

def get_homepage_and_similar_products_interaction_values(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    now = pd.Timestamp.now()
    sb = StringBuilder()
    sb.append(f'Interactions count: {len(raw_interactions)}\n')
    sb.append(f'Raw products not null fields count:\n{raw_products.count()}\n')

    adjust_raw_data(raw_interactions, raw_products)
    
    sb.append('\nOutlier detection\n')
    outlierConfig = OutlierDetectionConfig()
    raw_interactions, outlier_stats = outlier_detection(raw_interactions, outlierConfig)
    sb.append(get_outlier_detection_summary(outlier_stats, outlierConfig))
    sb.append('\n')

    raw_interactions['individual_rating'] = get_ratings_column_based_on_recency(now, raw_interactions)

    log_top_10_products(raw_interactions, sb, now)

    # Sorted by product_id
    grouped_interactions = raw_interactions.groupby(
        [PRODUCT_COL_NAME, USER_COL_NAME],
        observed=True # Mandatory, if we don't use this, we will get a lot of data from the moment when we casted column to categorical
    ).agg(
        total_rating=('individual_rating', 'sum'),
        interaction_count=('individual_rating', 'count')
    ).reset_index()

    grouped_interactions = apply_interaction_repeat_decay(grouped_interactions)

    sb.append(f'Grouped interactions of same users and products count: {len(grouped_interactions)}\n')

    # Filter users and products with interaction threshold
    user_interaction_counts = grouped_interactions.groupby(USER_COL_NAME, observed=True)['interaction_count'].sum()
    product_interaction_counts = grouped_interactions.groupby(PRODUCT_COL_NAME, observed=True)['interaction_count'].sum()

    filtered_interactions = get_threshold_filtered_interactions(grouped_interactions, user_interaction_counts, product_interaction_counts, sb)

    product_idx = filtered_interactions[PRODUCT_COL_NAME].cat.codes
    user_idx = filtered_interactions[USER_COL_NAME].cat.codes

    sparse_product_user_matrix = csr_matrix(
        (filtered_interactions['total_rating'], (product_idx, user_idx))
    )

    sparse_user_product_matrix = bm25_weight(sparse_product_user_matrix, K1=100, B=0.8).T.tocsr()

    product_ids = filtered_interactions[PRODUCT_COL_NAME].cat.categories
    user_ids = filtered_interactions[USER_COL_NAME].cat.categories
    sb.append(f'Unique product ids in interactions: {len(product_ids)}\n')
    sb.append(f'Unique user ids in interactions: {len(user_ids)}\n')

    raw_products_indexed = raw_products.set_index(ID_COL_NAME)

    missing_ids = pd.Index(product_ids).difference(raw_products_indexed.index)
    if not missing_ids.empty:
        sb.append(f"The product ids that were not found count: {len(missing_ids)}\n")
        sb.append(f"Products that were not found: {', '.join(map(str, missing_ids))}\n")

    products = raw_products_indexed.reindex(product_ids).reset_index(names=ID_COL_NAME) # Rows in the same order as sparse matrix columns
    default_values = { STOCK_COL_NAME: 0, STATUS_COL_NAME: 'Draft', TITLE_COL_NAME: 'Unknown Title' }
    products.fillna(value=default_values, inplace=True)

    # Build and attach PNG histogram image for email
    try:
        png_bytes = build_product_interaction_histogram_png(raw_interactions)
        attachments = [("product_interactions_hist.png", png_bytes, "image/png")]
    except Exception as ex:
        attachments = None
        sb.append(f"Failed to build histogram PNG: {ex}\n")

    sb.append(get_duration_message(now))
    Emailing().send_email_and_log_info(
        "Homepage and similar products recommender data cleaning",
        sb.__str__(),
        attachments=attachments,
        html=False
    )

    return sparse_user_product_matrix, user_ids.astype(str), products

# User worked 5 years for one company and was buying only one group of products, now he changed the company and want to buy other group of products, with this function we are forgetting previous interaction
def get_ratings_column_based_on_recency(now: pd.Timestamp, raw_interactions: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME], unit='s')
    diff_days = (now - timestamps) / np.timedelta64(1, 'D')

    if (diff_days < 0).any():
        raise ValueError("The timestamp is in the future. Please provide a valid past timestamp.")

    weights = raw_interactions[INTERACTION_COL_NAME].map(INTERACTION_WEIGHTS)

    if weights.isnull().any():
        raise ValueError("Interaction value doesn't exist (valid: Bought, PutInCart, PutInFavorites, Clicked).")

    # Apply interaction-specific decay rates
    # Each interaction type has its own decay scale for more nuanced recency modeling
    decayed_weights = weights * np.exp(-diff_days / INTERACTION_DECAY_SCALE)
    return decayed_weights

def apply_interaction_repeat_decay(raw_interactions: pd.DataFrame):
    decay_multiplier = INTERACTION_REPEAT_DECAY_FACTOR ** (raw_interactions['interaction_count'] - 1)
    raw_interactions['total_rating'] = raw_interactions['total_rating'] * decay_multiplier
    return raw_interactions

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
    
    sb.append('\nOutlier detection\n')
    outlierConfig = OutlierDetectionConfig()
    raw_interactions, outlier_stats = outlier_detection(raw_interactions, outlierConfig)
    sb.append(get_outlier_detection_summary(outlier_stats, outlierConfig))
    sb.append('\n')

    raw_interactions[TIMESTAMP_COL_NAME] = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME], unit='s')

    # Filter users and products with interaction threshold
    user_interaction_counts = raw_interactions.groupby(USER_COL_NAME, observed=True).size()
    product_interaction_counts = raw_interactions.groupby(PRODUCT_COL_NAME, observed=True).size()
    filtered_interactions = get_threshold_filtered_interactions(raw_interactions, user_interaction_counts, product_interaction_counts, sb)

    product_product_dataframe = get_product_product_dataframe(filtered_interactions)

    # product_product_dataframe.to_csv('../data/product_product.csv', index=False)
    # raise Exception()

    if product_product_dataframe.empty:
        raise BusinessException(f'There is no interactions between any of the products within the same user in one {SESSION_HOURS}h session period.')

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
    default_values = {STOCK_COL_NAME: 0, STATUS_COL_NAME: 'Draft', TITLE_COL_NAME: 'Unknown Title' }
    products.fillna(value=default_values, inplace=True)

    # Don't apply BM25 weighting for cross-sell - it's not appropriate for product-product similarity
    # clean_sparse_interactions = bm25_weight(clean_sparse_interactions).tocsr()

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

    decayed_weights = weights * np.exp(-((days_diff / INTERACTION_DECAY_SCALE) ** 2)) # Faster reduce in first couple of days but as days increase reduce is getting slower and slower
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

    Emailing().send_email(Settings().EXCEPTION_EMAILS, 'Exception occurred in pa-recommender', traceback.format_exc())

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

    # Pre-cast to category for faster grouping and later extraction of codes
    raw_interactions[PRODUCT_COL_NAME] = raw_interactions[PRODUCT_COL_NAME].astype(int).astype('category')
    
    # NOTE: If we switch the logic back to user_id, uncomment this code
    # CB API returns some IDs as strings; convert to int for consistency 
    # raw_interactions[USER_COL_NAME] = raw_interactions[USER_COL_NAME].astype(int).astype('category') 
    raw_interactions[USER_COL_NAME] = raw_interactions[USER_COL_NAME].astype('category')

def get_duration_message(start_time: pd.Timestamp) -> str:
    return f'Duration: {(pd.Timestamp.now() - start_time).seconds} seconds'

def get_interactions_from_external_api():
    now = pd.Timestamp.now()
    sb = StringBuilder()

    base_url = (
        f"{Settings().API_URL}/GET/events/"
        f"?namespace={EXTERNAL_API_NAMESPACE}"
        f"&order_by=id&order_by_type=desc"
    )
    
    limit_range = 10000

    events = ['add_to_cart', 'initiate_checkout', 'purchase', 'add_to_wishlist', 'content_view']
    # Keep independent pagination state per event to avoid redundant calls once an event is exhausted
    event_state = {event: {"offset": 0, "done": False} for event in events}

    one_year_ago = datetime.utcnow() - timedelta(days=100)
    one_year_ago_unix_timestamp = int(one_year_ago.timestamp())

    data_dir = '../data'
    cache_path = os.path.join(data_dir, 'filtered_interactions.csv')

    # Load existing filtered interactions from cache
    existing_interactions = pd.DataFrame()
    try:
        existing_interactions = pd.read_csv(cache_path)
        if not existing_interactions.empty:
            # Keep only last year
            existing_interactions = existing_interactions.loc[existing_interactions['created'] >= one_year_ago_unix_timestamp]
            last_activity_date = int(existing_interactions['created'].max()) if not existing_interactions.empty else one_year_ago_unix_timestamp
            sb.append(f"Loaded {len(existing_interactions)} existing filtered interactions from cache.\n")
        else:
            last_activity_date = one_year_ago_unix_timestamp
    except Exception as ex:
        sb.append(f"Failed reading cache: {ex}\n")
        last_activity_date = one_year_ago_unix_timestamp

    # Fetch new data
    new_filtered_interactions = []

    # Continue until every event is marked done
    while any(not s["done"] for s in event_state.values()):
        for event in events:
            state = event_state[event]
            if state["done"]:
                continue

            start = state["offset"]
            end = start + limit_range
            sb.append(f"Event: {event}. Fetching raw data from: {start} to: {end}\n")
            print(f"Event: {event}. Fetching raw data from: {start} to: {end}\n")

            url = (
                f"{base_url}"
                f"&date_filter_from={last_activity_date}"
                f"&event={event}"
                f"&limit={start},{limit_range}"
            )

            response = requests.get(url, headers=EXTERNAL_API_HEADERS)
            if response.status_code != 200:
                raise BusinessException(f"External CB request failed: {response.status_code} {response.text}.\n")

            json_payload = response.json()
            data_section = json_payload.get("data", {})
            batch_activities = data_section.get("activities", []) if data_section else []

            if not batch_activities:
                state["done"] = True # No more data for this event; mark as done
                continue

            df_batch = pd.DataFrame(batch_activities)
            batch_filtered = get_filtered_interactions(df_batch)
            
            if not batch_filtered.empty:
                new_filtered_interactions.append(batch_filtered)

            state["offset"] = end
            time.sleep(0.05) # Tiny delay to avoid hammering the API

    # Combine existing and new filtered interactions
    all_filtered_interactions = existing_interactions.copy()
    
    if new_filtered_interactions:
        new_df = pd.concat(new_filtered_interactions, ignore_index=True)
        all_filtered_interactions = pd.concat([all_filtered_interactions, new_df], ignore_index=True)

        # Drop duplicates
        count_before_duplicates = len(all_filtered_interactions)
        all_filtered_interactions = all_filtered_interactions.drop_duplicates(subset='id').reset_index(drop=True)
        count_after_duplicates = len(all_filtered_interactions)
        sb.append(f'Dropped {count_before_duplicates - count_after_duplicates} duplicates.\n')

    if all_filtered_interactions.empty:
        raise BusinessException("Interactions are required.")

    # Save cache
    os.makedirs(data_dir, exist_ok=True)
    all_filtered_interactions.to_csv(cache_path, index=False)
    sb.append(f"Cache updated with {len(all_filtered_interactions)} filtered interactions.\n")

    sb.append(get_duration_message(now))
    Emailing().send_email_and_log_info("Getting interactions from CB API", sb.__str__())

    return all_filtered_interactions

def get_filtered_interactions(activities_df):
    """Process raw activities into filtered interactions"""
    if activities_df.empty:
        return pd.DataFrame()
    
    dict_info_mask = (
        activities_df["info"].notna() &
        (activities_df["info"].map(type) == dict) & 
        (activities_df["info"] != {}) 
    )

    filtered_interactions = (
        activities_df
        .loc[dict_info_mask, ['id', 'action', USER_COL_NAME, 'info', 'created']]
        .dropna(subset=[USER_COL_NAME])
        .copy()
    )

    # filtered_interactions[USER_COL_NAME] = filtered_interactions[USER_COL_NAME].astype(int) # NOTE: If we switch the logic back to user_id, uncomment this code

    filtered_interactions = manipulate_action_with_content_ids(filtered_interactions, 'initiate_checkout')
    filtered_interactions = manipulate_action_with_content_ids(filtered_interactions, 'purchase')
    filtered_interactions = manipulate_action_with_content_ids(filtered_interactions, 'add_to_wishlist')

    filtered_interactions = manipulate_action_with_product_id(filtered_interactions, 'add_to_cart')
    filtered_interactions = manipulate_action_with_product_id(filtered_interactions, 'content_view')

    filtered_interactions = filtered_interactions[
        (filtered_interactions['product_id'].notnull()) & 
        (filtered_interactions['product_id'] != '') & 
        (filtered_interactions['product_id'] != 'nan') # CB has 'nan' values for product_id so we need to do this
    ].reset_index(drop=True)

    filtered_interactions = filtered_interactions.drop(columns=['info'])

    return filtered_interactions

def manipulate_action_with_content_ids(interactions: pd.DataFrame, action_name: str):
    if interactions is None or interactions.empty:
        return interactions
    
    mask = interactions['action'] == action_name

    if not mask.any():
        return interactions
    
    df = interactions.loc[mask].copy()
    temp_info_df = pd.DataFrame(df['info'].tolist(), index=df.index)

    df['content_ids'] = temp_info_df.get('content_ids', pd.Series("", index=df.index)).astype(str)

    df = df[df['content_ids'] != '']

    if df.empty:
        return interactions

    df['product_id'] = df['content_ids'].str.strip(',').str.split(',')

    exploded = df.explode('product_id').reset_index(drop=True)
    exploded = exploded.drop(columns=['content_ids'])

    others = interactions.loc[~mask].copy().reset_index(drop=True)
    interactions = pd.concat([others, exploded], ignore_index=True)

    return interactions

def manipulate_action_with_product_id(interactions: pd.DataFrame, action_name: str):
    """
    Process actions that have direct product_id (like add_to_cart, content_view)
    These have a single product ID per interaction
    """
    if interactions is None or interactions.empty:
        return interactions
    
    mask = interactions['action'] == action_name

    if not mask.any():
        return interactions
    
    info = interactions.loc[mask, 'info'].tolist()
    temp_info_df = pd.DataFrame(info, index=interactions.loc[mask].index)

    interactions.loc[mask, 'product_id'] = temp_info_df['id']

    # Handle content_view specific filtering
    if action_name == 'content_view':
        # Find rows where referrer is missing
        missing_referrer_mask = temp_info_df['referrer'].isna()
        
        if missing_referrer_mask.any():
            indices_to_drop = temp_info_df.index[missing_referrer_mask]
            interactions = interactions.drop(indices_to_drop)
            print(f"Dropped {len(indices_to_drop)} content_view rows with missing referrer")
        
    return interactions

def get_products_from_external_api():
    now = pd.Timestamp.now()
    sb = StringBuilder()

    base_url = (
        f"{Settings().API_URL}/GET/products/short/"
        f"?namespace={EXTERNAL_API_NAMESPACE}"
    )

    limit_from = 0
    limit_range = 10_000

    all_products = []  # will hold dicts from each batch

    while True:
        sb.append(f"Fetching raw data from: {limit_from} to: {limit_from + limit_range}\n")

        url = (
            f"{base_url}"
            f"&limit={limit_from},{limit_range}"
        )

        response = requests.get(url, headers=EXTERNAL_API_HEADERS)

        if response.status_code == 200:
            json_payload = response.json()
            data_section = json_payload.get("data", {})
            batch_products = data_section.get("products", []) if data_section else []
            
            if batch_products == []:
                break

            all_products.extend(batch_products)
        else:
            raise BusinessException(f"External CB request failed: {response}.\n")

        limit_from = limit_from + limit_range

    if not all_products:
        raise BusinessException("Products are required.")

    new_raw_products = pd.DataFrame(all_products)

    filtered_products = new_raw_products[['id', 'stock', 'status', 'title']].copy()

    sb.append(get_duration_message(now))
    Emailing().send_email_and_log_info("Getting products from CB API", sb.__str__())

    return filtered_products

# This method doesn't handle stock==0 filtering, this is not the same data as in top_10_products_to_recommend, it's just made for intuitive purposes
def log_top_10_products(raw_interactions: pd.DataFrame, processingLog: StringBuilder, now: pd.Timestamp):
    # Log Top 10 products by total rating with interaction breakdown
    try:
        product_totals = (
            raw_interactions
                .groupby(PRODUCT_COL_NAME, observed=True)['individual_rating']
                .sum()
                .sort_values(ascending=False)
        )

        top_products = product_totals.head(10)

        if not top_products.empty:
            processingLog.append('\nTop products by total rating (with interaction breakdown):\n')
            for product_id, total in top_products.items():
                processingLog.append(f"\nProduct {int(product_id)} | total_rating={total:.6f}\n")

                product_rows = raw_interactions[raw_interactions[PRODUCT_COL_NAME] == product_id].copy()

                # Compute helpful explainer columns (without mutating original dtypes irreversibly)
                timestamps_dt = pd.to_datetime(product_rows[TIMESTAMP_COL_NAME], unit='s')
                days_ago = (now - timestamps_dt) / np.timedelta64(1, 'D')
                base_weight = product_rows[INTERACTION_COL_NAME].map(INTERACTION_WEIGHTS)
                decayed = product_rows['individual_rating']

                # Sort newest first for readability
                order = timestamps_dt.sort_values(ascending=False).index[:5]
                for idx in order:
                    uid = product_rows.at[idx, USER_COL_NAME]
                    action = product_rows.at[idx, INTERACTION_COL_NAME]
                    created_unix = product_rows.at[idx, TIMESTAMP_COL_NAME]
                    created_iso = pd.to_datetime(created_unix, unit='s').isoformat()
                    processingLog.append(
                        f"  - user={uid}, action={action}, created={created_iso}, "
                        f"base_weight={base_weight.at[idx]:.3f}, days_ago={days_ago.at[idx]:.2f}, "
                        f"decay_scale={INTERACTION_DECAY_SCALE:.2f}, decayed_rating={decayed.at[idx]:.6f}\n"
                    )
    except Exception as ex:
        # Do not fail on logging issues
        processingLog.append(f"Failed building top-products interaction log: {ex}\n")

def get_threshold_filtered_interactions(interactions: pd.DataFrame, user_interaction_counts: pd.DataFrame, product_interaction_counts: pd.DataFrame, processingLog: StringBuilder) -> pd.DataFrame:
    valid_users, valid_products = get_users_and_products_above_interaction_threshold(user_interaction_counts, product_interaction_counts, processingLog)

    # Apply filters to interactions
    filtered_interactions = interactions[
        (interactions[USER_COL_NAME].isin(valid_users)) &
        (interactions[PRODUCT_COL_NAME].isin(valid_products))
    ].copy()
    
    # Remove unused categories after filtering
    filtered_interactions[USER_COL_NAME] = filtered_interactions[USER_COL_NAME].cat.remove_unused_categories()
    filtered_interactions[PRODUCT_COL_NAME] = filtered_interactions[PRODUCT_COL_NAME].cat.remove_unused_categories()
    
    processingLog.append(f'Interactions after threshold filtering: {len(filtered_interactions)}\n')

    return filtered_interactions

def get_users_and_products_above_interaction_threshold(user_interaction_counts: pd.DataFrame, product_interaction_counts: pd.DataFrame, processingLog: StringBuilder) -> tuple[pd.Index, pd.Index]:
    # Filter users with interaction threshold
    valid_users = user_interaction_counts[user_interaction_counts > 2].index
    processingLog.append(f'{len(valid_users)} after threshold filtering\n')
    
    # Filter products with interaction threshold
    valid_products = product_interaction_counts[product_interaction_counts > 4].index
    processingLog.append(f'{len(valid_products)} after threshold filtering\n')

    return valid_users, valid_products

#endregion

#region Visualization helpers

def build_product_interaction_histogram_png(
    raw_interactions: pd.DataFrame,
    bins: int = 50,
    by_action: str | list[str] | None = None
) -> bytes:
    if by_action is not None:
        if isinstance(by_action, str):
            mask = raw_interactions[INTERACTION_COL_NAME] == by_action
        else:
            mask = raw_interactions[INTERACTION_COL_NAME].isin(by_action)
        data = raw_interactions.loc[mask]
    else:
        data = raw_interactions

    if data.empty:
        raise BusinessException("No interactions available for plotting after filtering.")

    counts_per_product = data.groupby(PRODUCT_COL_NAME, observed=True).size()
    if counts_per_product.empty:
        raise BusinessException("Grouping produced no data; verify input columns and types.")

    counts = counts_per_product.values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
    ax.hist(counts, bins=bins, color="#4e79a7")
    ax.set_xlabel("Interactions per product")
    ax.set_ylabel("Number of products")
    ax.set_title("Distribution of interactions per product")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

#endregion