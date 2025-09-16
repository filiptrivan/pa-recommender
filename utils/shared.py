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
from datetime import datetime, timedelta
from utils.outlier_detection import outlier_detection, get_outlier_detection_summary
from utils.outlier_config import OutlierDetectionConfig

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
INTERACTION_DECAY_SCALES = {
    'purchase': 30,           # Purchases stay relevant longer
    'initiate_checkout': 20,  # Checkout attempts moderately relevant
    'add_to_cart': 15,        # Cart additions decay faster
    'add_to_wishlist': 10,    # Wishlist items decay quickly
    'content_view': 5         # Views decay fastest
}

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
    
    raw_interactions, outlier_stats = outlier_detection(raw_interactions, OutlierDetectionConfig())
    
    sb.append(get_outlier_detection_summary(outlier_stats))
    sb.append('\n')

    raw_interactions['individual_rating'] = get_ratings_column_based_on_recency(now, raw_interactions)

    log_top_10_products(raw_interactions, sb, now)

    sb.append(email_product_interaction_histogram_html(raw_interactions))

    # Sorted by product_id
    grouped_interactions = raw_interactions.groupby(
        [PRODUCT_COL_NAME, USER_COL_NAME],
        observed=True # Mandatory, if we don't use this, we will get a lot of data from the moment when we casted column to categorical
    ).agg(
        total_rating=('individual_rating', 'sum'),
        interaction_count=('individual_rating', 'count')
    ).reset_index()

    sb.append(f'Grouped interactions of same users and products count: {len(grouped_interactions)}\n')

    # Filter users with interaction threshold
    user_interaction_counts = grouped_interactions.groupby(USER_COL_NAME, observed=True)['interaction_count'].sum()
    valid_users = user_interaction_counts[user_interaction_counts >= 2].index
    sb.append(f'Users: {len(user_interaction_counts)} total; {len(valid_users)} after filtering; {len(user_interaction_counts) - len(valid_users)} filtered out\n')
    
    # Filter products with interaction threshold
    product_interaction_counts = grouped_interactions.groupby(PRODUCT_COL_NAME, observed=True)['interaction_count'].sum()
    valid_products = product_interaction_counts[product_interaction_counts >= 4].index
    sb.append(f'Products: {len(product_interaction_counts)} total; {len(valid_products)} after filtering; {len(product_interaction_counts) - len(valid_products)} filtered out\n')
    
    # Apply filters to grouped_interactions
    filtered_interactions = grouped_interactions[
        (grouped_interactions[USER_COL_NAME].isin(valid_users)) &
        (grouped_interactions[PRODUCT_COL_NAME].isin(valid_products))
    ].copy()
    
    # Remove unused categories after filtering
    filtered_interactions[USER_COL_NAME] = filtered_interactions[USER_COL_NAME].cat.remove_unused_categories()
    filtered_interactions[PRODUCT_COL_NAME] = filtered_interactions[PRODUCT_COL_NAME].cat.remove_unused_categories()
    
    sb.append(f'Interactions after filtering: {len(filtered_interactions)}\n')
    sb.append(f'Interactions filtered out: {len(grouped_interactions) - len(filtered_interactions)}\n')

    product_idx = filtered_interactions[PRODUCT_COL_NAME].cat.codes
    user_idx = filtered_interactions[USER_COL_NAME].cat.codes

    clean_sparse_interactions = csr_matrix(
        (filtered_interactions['total_rating'], (user_idx, product_idx))
    )

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

    sb.append(get_duration_message(now))
    Emailing().send_email_and_log_info("Homepage and similar products recommender data cleaning", sb.__str__())

    return clean_sparse_interactions, user_ids.astype(str), products

# User worked 5 years for one company and was buying only one group of products, now he changed the company and want to buy other group of products, with this function we are forgetting previous interaction
def get_ratings_column_based_on_recency(now: pd.Timestamp, raw_interactions: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME], unit='s')
    diff_days = (now - timestamps) / np.timedelta64(1, 'D')

    if (diff_days < 0).any():
        raise ValueError("The timestamp is in the future. Please provide a valid past timestamp.")

    weights = raw_interactions[INTERACTION_COL_NAME].map(INTERACTION_WEIGHTS)
    decay_scales = raw_interactions[INTERACTION_COL_NAME].map(INTERACTION_DECAY_SCALES)

    if weights.isnull().any():
        raise ValueError("Interaction value doesn't exist (valid: Bought, PutInCart, PutInFavorites, Clicked).")
    
    if decay_scales.isnull().any():
        raise ValueError("Interaction decay scale doesn't exist for all interaction types.")

    # Apply interaction-specific decay rates
    # Each interaction type has its own decay scale for more nuanced recency modeling
    decayed_weights = weights * np.exp(-diff_days / decay_scales)
    return decayed_weights

def bm25_weight(X, K1=2.0, B=0.25):
    """BM25 weighting for sparse user-item matrix"""
    X = coo_matrix(X, copy=True)
    
    N = float(X.shape[0])  # total users
    df = np.bincount(X.col)  # number of users per product
    idf = np.log((N - df + 0.5) / (df + 0.5))
    
    col_sums = np.ravel(X.sum(axis=0))
    avg_len = col_sums.mean()
    length_norm = (1 - B) + B * col_sums / avg_len
    
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.col] + X.data) * idf[X.col]
    return X.tocsr()

#endregion

#region Exploratory HTML reports

def email_product_interaction_histogram_html(
    raw_interactions: pd.DataFrame,
    bins: int = 50,
    by_action: str | list[str] | None = None,
    subject_prefix: str = "Product interaction histogram"
) -> None:
    """Compute interactions-per-product distribution and email an HTML report with inline SVG histogram.

    Uses the standard logging email channel. No files are written.
    """
    # Filter by action if requested
    if by_action is not None:
        if isinstance(by_action, str):
            mask = raw_interactions[INTERACTION_COL_NAME] == by_action
            title_suffix = f" (action={by_action})"
        else:
            mask = raw_interactions[INTERACTION_COL_NAME].isin(by_action)
            title_suffix = f" (actions={','.join(map(str, by_action))})"
        data = raw_interactions.loc[mask]
    else:
        data = raw_interactions
        title_suffix = ""

    if data.empty:
        raise BusinessException("No interactions available for plotting after filtering.")

    counts_per_product = data.groupby(PRODUCT_COL_NAME, observed=True).size()
    if counts_per_product.empty:
        raise BusinessException("Grouping produced no data; verify input columns and types.")

    counts = counts_per_product.values.astype(float)
    hist, bin_edges = np.histogram(counts, bins=bins)

    # Basic stats
    num_products = int(len(counts))
    mean_c = float(np.mean(counts))
    median_c = float(np.median(counts))
    max_c = int(np.max(counts))

    # Build simple inline SVG histogram
    width, height = 800, 320
    margin_left, margin_right, margin_top, margin_bottom = 50, 20, 20, 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_y = max(1, int(hist.max()))
    num_bins = len(hist)
    bar_w = plot_w / max(1, num_bins)

    # Y-axis ticks (0, max)
    svg_bars = []
    for i, y in enumerate(hist):
        bar_height = 0 if max_y == 0 else (y / max_y) * plot_h
        x = margin_left + i * bar_w
        y_top = margin_top + (plot_h - bar_height)
        svg_bars.append(
            f'<rect x="{x:.2f}" y="{y_top:.2f}" width="{bar_w - 1:.2f}" height="{bar_height:.2f}" fill="#4e79a7" />'
        )

    # Axes
    x_axis = f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333" stroke-width="1" />'
    y_axis = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333" stroke-width="1" />'

    # Y tick labels (0 and max)
    y0_label = f'<text x="{margin_left - 10}" y="{margin_top + plot_h}" text-anchor="end" font-size="10" fill="#333">0</text>'
    ymax_label = f'<text x="{margin_left - 10}" y="{margin_top + 10}" text-anchor="end" font-size="10" fill="#333">{max_y}</text>'

    # Title and axis labels
    title = f'Distribution of interactions per product{title_suffix}'
    title_svg = f'<text x="{width/2}" y="{margin_top - 5}" text-anchor="middle" font-size="14" fill="#111">{title}</text>'
    x_label = 'Interactions per product (binned)'
    y_label = 'Number of products'
    x_label_svg = f'<text x="{margin_left + plot_w/2}" y="{height - 5}" text-anchor="middle" font-size="12" fill="#333">{x_label}</text>'
    y_label_svg = f'<text transform="translate(15,{margin_top + plot_h/2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#333">{y_label}</text>'

    svg = (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'{title_svg}'
        f'{x_axis}{y_axis}'
        f"{''.join(svg_bars)}"
        f'{y0_label}{ymax_label}'
        f'{x_label_svg}{y_label_svg}'
        f'</svg>'
    )

    # Build HTML body
    html_stats = (
        f"<p><strong>Products:</strong> {num_products} | "
        f"<strong>Mean:</strong> {mean_c:.2f} | "
        f"<strong>Median:</strong> {median_c:.2f} | "
        f"<strong>Max:</strong> {max_c}</p>"
    )
    html = f"""
<h3>{title}</h3>
{html_stats}
{svg}
"""

    return html

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
    
    # Outlier detection
    sb.append('\nOutlier detection\n')
    raw_interactions, outlier_stats = outlier_detection(raw_interactions, OutlierDetectionConfig())
    sb.append(get_outlier_detection_summary(outlier_stats))
    sb.append('\n')

    raw_interactions[TIMESTAMP_COL_NAME] = pd.to_datetime(raw_interactions[TIMESTAMP_COL_NAME], unit='s')

    # Filter users with interaction threshold
    user_interaction_counts = raw_interactions.groupby(USER_COL_NAME, observed=True).size()
    valid_users = user_interaction_counts[user_interaction_counts >= 3].index
    sb.append(f'Users before filtering: {len(user_interaction_counts)}\n')
    sb.append(f'Users after threshold filtering: {len(valid_users)}\n')
    sb.append(f'Users filtered out: {len(user_interaction_counts) - len(valid_users)}\n')
    
    # Filter products with interaction threshold
    product_interaction_counts = raw_interactions.groupby(PRODUCT_COL_NAME, observed=True).size()
    valid_products = product_interaction_counts[product_interaction_counts >= 5].index
    sb.append(f'Products before filtering: {len(product_interaction_counts)}\n')
    sb.append(f'Products after threshold filtering: {len(valid_products)}\n')
    sb.append(f'Products filtered out: {len(product_interaction_counts) - len(valid_products)}\n')
    
    # Apply filters to raw_interactions
    filtered_interactions = raw_interactions[
        (raw_interactions[USER_COL_NAME].isin(valid_users)) &
        (raw_interactions[PRODUCT_COL_NAME].isin(valid_products))
    ].copy()
    
    # Remove unused categories after filtering
    filtered_interactions[USER_COL_NAME] = filtered_interactions[USER_COL_NAME].cat.remove_unused_categories()
    filtered_interactions[PRODUCT_COL_NAME] = filtered_interactions[PRODUCT_COL_NAME].cat.remove_unused_categories()
    
    sb.append(f'Interactions after filtering: {len(filtered_interactions)}\n')
    sb.append(f'Interactions filtered out: {len(raw_interactions) - len(filtered_interactions)}\n')

    product_product_dataframe = get_product_product_dataframe(filtered_interactions)

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
    decay_scales = merged_interactions[f"{INTERACTION_COL_NAME}{RIGHT_SUFFIX}"].map(INTERACTION_DECAY_SCALES).values

    # Apply interaction-specific decay rates for cross-sell recommendations
    # Using linear decay (not quadratic) for consistency with homepage recommendations
    decayed_weights = weights * np.exp(-days_diff / decay_scales)
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
    
    limit_from = 0
    limit_range = 10000

    events = ['add_to_cart', 'initiate_checkout', 'purchase', 'add_to_wishlist', 'content_view']

    one_year_ago = datetime.utcnow() - timedelta(days=10)
    one_year_ago_unix_timestamp = int(one_year_ago.timestamp())

    all_activities = []  # Will hold dicts from each batch

    while True:
        all_empty = True  # Will track if all events returned empty batches this iteration

        for event in events:
            sb.append(f"Event: {event}. Fetching raw data from: {limit_from} to: {limit_from + limit_range}\n")
            print(f"Event: {event}. Fetching raw data from: {limit_from} to: {limit_from + limit_range}\n")

            url = (
                f"{base_url}"
                f"&date_filter_from={one_year_ago_unix_timestamp}"
                f"&event={event}"
                f"&limit={limit_from},{limit_range}"
            )

            response = requests.get(url, headers=EXTERNAL_API_HEADERS)

            if response.status_code == 200:
                json_payload = response.json()
                data_section = json_payload.get("data", {})
                batch_activities = data_section.get("activities", []) if data_section else []

                if batch_activities != []:
                    all_activities.extend(batch_activities)
                    all_empty = False # At least one event had data this iteration
            else:
                raise BusinessException(f"External CB request failed: {response}.\n")
        if all_empty:
            break # All events returned empty lists, stop fetching more

        limit_from = limit_from + limit_range

    if not all_activities:
        raise BusinessException("Interactions are required.")

    new_raw_interactions = pd.DataFrame(all_activities)

    dict_info_mask = (new_raw_interactions["info"].map(type) == dict) & \
        (new_raw_interactions["info"].map(len) > 0)

    filtered_interactions = (
        new_raw_interactions
        .loc[dict_info_mask, ['action', USER_COL_NAME, 'info', 'created']]
        .dropna(subset=[USER_COL_NAME, 'info'])
        .copy()
    )

    # filtered_interactions[USER_COL_NAME] = filtered_interactions[USER_COL_NAME].astype(int) # NOTE: If we switch the logic back to user_id, uncomment this code

    filtered_interactions = manipulate_action_with_content_ids(filtered_interactions, 'initiate_checkout')
    filtered_interactions = manipulate_action_with_content_ids(filtered_interactions, 'purchase')
    filtered_interactions = manipulate_action_with_content_ids(filtered_interactions, 'add_to_wishlist')

    manipulate_action_with_product_id(filtered_interactions, 'add_to_cart')
    manipulate_action_with_product_id(filtered_interactions, 'content_view')

    filtered_interactions = filtered_interactions[
        (filtered_interactions['product_id'].notnull()) & 
        (filtered_interactions['product_id'] != '') & 
        (filtered_interactions['product_id'] != 'nan') # CB has 'nan' values for product_id so we need to do this
    ].reset_index(drop=True)

    filtered_interactions = filtered_interactions.drop(columns=['info'])

    sb.append(get_duration_message(now))
    Emailing().send_email_and_log_info("Getting interactions from CB API", sb.__str__())

    return filtered_interactions

def manipulate_action_with_content_ids(interactions: pd.DataFrame, action_name: str):
    mask = interactions['action'] == action_name
    df = interactions.loc[mask].copy()
    temp_info_df = pd.DataFrame(df['info'].tolist(), index=df.index)

    df['content_ids'] = temp_info_df.get('content_ids', pd.Series("", index=df.index)).astype(str)

    df = df[df['content_ids'] != '']

    df['product_id'] = df['content_ids'].str.strip(',').str.split(',')

    exploded = df.explode('product_id').reset_index(drop=True)
    exploded = exploded.drop(columns=['content_ids'])

    others = interactions.loc[~mask].copy().reset_index(drop=True)
    interactions = pd.concat([others, exploded], ignore_index=True)

    return interactions

def manipulate_action_with_product_id(interactions: pd.DataFrame, action_name: str):
    mask = interactions['action'] == action_name
    info = interactions.loc[mask, 'info'].tolist()
    temp_info_df = pd.DataFrame(info, index=interactions.loc[mask].index)
    interactions.loc[mask, 'product_id'] = temp_info_df['id']

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
                decay_scale = product_rows[INTERACTION_COL_NAME].map(INTERACTION_DECAY_SCALES)
                decayed = product_rows['individual_rating']

                # Sort newest first for readability
                order = timestamps_dt.sort_values(ascending=False).index
                for idx in order:
                    uid = product_rows.at[idx, USER_COL_NAME]
                    action = product_rows.at[idx, INTERACTION_COL_NAME]
                    created_unix = product_rows.at[idx, TIMESTAMP_COL_NAME]
                    created_iso = pd.to_datetime(created_unix, unit='s').isoformat()
                    processingLog.append(
                        f"  - user={uid}, action={action}, created={created_iso}, "
                        f"base_weight={base_weight.at[idx]:.3f}, days_ago={days_ago.at[idx]:.2f}, "
                        f"decay_scale={decay_scale.at[idx]:.2f}, decayed_rating={decayed.at[idx]:.6f}\n"
                    )
    except Exception as ex:
        # Do not fail on logging issues
        processingLog.append(f"Failed building top-products interaction log: {ex}\n")

#endregion