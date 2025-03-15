from functools import wraps
import logging
import os
import io
import csv
import json
from flask import Response, jsonify, request
import pandas as pd
from collections import defaultdict
import numpy as np
from math import exp
from math import log1p
from azure.storage.blob import BlobServiceClient
from exceptions.BusinessException import BusinessException

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#region Data Manipulation

PRODUCT_COL_NAME = 'ProductId'
USER_COL_NAME = 'UserId'
INTERACTION_COL_NAME = 'Interaction'

RECENCY_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.
MULTIPLE_INTERACTIONS_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.

def save_interaction_values(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    now = pd.Timestamp.now()
    pivot_data = defaultdict(dict)
    grouped_ratings = defaultdict(list)

    for _, row in raw_interactions.iterrows():
        key = (str(row[PRODUCT_COL_NAME]), row[USER_COL_NAME])
        grouped_ratings[key].append(row)

    for (product_id, user_id), rows in grouped_ratings.items():
        rating = 0
        
        for i in range(len(rows)):
            row = rows[i]

            # FT: Recency bonus
            timestamp = parse_and_format_timestamp(row['Timestamp'])
            diff_days  = (now - timestamp).total_seconds() / (60 * 60 * 24)

            if diff_days < 0:
                raise ValueError("The timestamp is in the future; please provide a valid past timestamp.")

            interaction = row[INTERACTION_COL_NAME]
            if interaction == 'Bought':
                rating += get_rating_based_on_recency(diff_days, 1)
            elif interaction == 'PutInCart':
                rating += get_rating_based_on_recency(diff_days, 0.5)
            elif interaction == 'PutInFavorites':
                rating += get_rating_based_on_recency(diff_days, 0.3)
            elif interaction == 'Clicked':
                rating += get_rating_based_on_recency(diff_days, 0.1)
            else:
                raise ValueError("Interaction value doesn't exist.")

        rating += get_multiple_interaction_bonus(len(rows), rating)

        pivot_data[product_id][user_id] = None if rating == 0 else rating

    product_ids = sorted(pivot_data.keys())
    user_ids = sorted(raw_interactions[USER_COL_NAME].unique())
    
    # FT: Needs to use list for products instead of numpy array because it contains different data types
    products = []
    users = []
    clean_interactions = []

    for user_id in user_ids:
        users.append(user_id)

    for product_id in product_ids:
        row = [pivot_data[product_id].get(user_id, '') for user_id in user_ids]
        product_df: pd.DataFrame = raw_products.loc[raw_products['SKU'] == product_id]

        if not product_df.empty:
            product = product_df.iloc[0]
            stock = int(product.get('Stock', 0))
            status = product.get('Status', 'Draft')
            visibility = product.get('Visibility', 'Private')
            active = bool(product.get('Active', False))
        else:
            stock, status, visibility, active = 0, 'Draft', 'Private', False

        products.append([product_id, stock, status, visibility, active])
        clean_interactions.append(row)
    
    return clean_interactions, products, users

# Maybe im not happy with the product that i bought only one time, so for example 2 clicks and 1 put in favorites is stronger than that
# When i buy product a lot of times other products couldn't ever be recommended, so the max for this bonus is 1
def get_multiple_interaction_bonus(num_of_interactions, rating):
    if num_of_interactions == 1:
        return 0
    
    # bonus = 1 - (1 / (1 + (num_of_interactions - 1) * rating)) # FT: Slower rise and bigger initial values
    bonus = 1 - (1 / (1 + log1p((num_of_interactions - 1) * rating))) # FT: Slower rise and lower initial values

    return min(bonus, 1)

# User worked 5 years for one company and was buying only one group of products, now he changed the company and want to buy other group of products, with this function we are forgetting previous interaction
def get_rating_based_on_recency(diff_days, interaction_weight, decay_scale=RECENCY_DECAY_SCALE):
    return interaction_weight * exp(-diff_days / decay_scale) # FT: Faster reduce in first couple of days but as days increase reduce is getting slower and slower

def test_recency_bonus(days, interaction_weight, decay_scale=RECENCY_DECAY_SCALE):
    first_days = {day: get_rating_based_on_recency(day, interaction_weight, decay_scale) for day in range(1, days)}

    for day, value in first_days.items():
        logger.info(f"Day {day}: {value:.4f}")

def parse_and_format_timestamp(value): 
    if isinstance(value, str):
        timestamp = pd.to_datetime(value, format='%d/%m/%Y')
    else:
        timestamp = switch_months_and_days(value)
    
    return timestamp

def switch_months_and_days(value):
    """
    Correct a date that may have been parsed using a MM/DD/YYYY format
    when it should be interpreted as DD/MM/YYYY.
    
    If the day component (currently in the month position) is 12 or less,
    swap the month and day. Otherwise, return the original timestamp.
    """
    dt = pd.Timestamp(value)
    if dt.day <= 12:
        try:
            corrected = pd.Timestamp(year=dt.year, month=dt.day, day=dt.month)
            return corrected
        except ValueError:
            return dt
    else:
        return dt

def get_dense_interactions_matrix(clean_interactions: list):
    arr = np.array(clean_interactions)
    
    def convert(val):
        if val == '' or val is None:
            return 0.0
        return float(val)

    convert_vec = np.vectorize(convert)
    Y = convert_vec(arr)
    
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
        exception = str(ex)

    if isinstance(ex, BusinessException):
        code = ex.code
        log_level = logging.WARN
        message = ex.message
    else:
        code = 500
        log_level = logging.ERROR
        message = "An error occurred in the system, our team has been informed and will fix it as soon as possible. Thank you for your patience."

    logger.log(log_level, str(ex))

    response = Response(
        response=json.dumps({
            "message": message,
            "exception": exception,
        }),
        status=code,
        mimetype='application/json'
    )

    return response

def load_csv_dict(filepath):
    """Load CSV data from a file and return a list of rows."""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return [row for row in reader]

def load_csv_list(filepath):
    """Load CSV data from a file and return a list of rows."""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        return [row for row in reader]
    
def load_csv_df(filepath, header=None):
    return pd.read_csv(filepath, delimiter=';', header=header)
    
def load_excel_list(filepath):
    """Load Excel data from a file and return a list of rows as dictionaries."""
    df = pd.read_excel(filepath)
    return df.to_dict(orient='records')

def load_excel_from_azure(file_name):
    file_stream = load_file_stream_from_azure(file_name)
    df = pd.read_csv(io.BytesIO(file_stream))
    return df

def load_dict_from_azure(file_name):
    file_stream = load_file_stream_from_azure(file_name)

    if file_stream is None:
        return None

    dictionary = json.load(io.BytesIO(file_stream))
    return dictionary

def load_file_stream_from_azure(file_name):
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    container_client = blob_service_client.get_container_client(os.getenv('CONTAINER_NAME'))
    blob_client = container_client.get_blob_client(file_name)

    try:
        stream = blob_client.download_blob()
    except:
        return None

    file_stream = stream.readall()

    return file_stream

def save_dictionary_to_azure(file_name, dictionary: dict):
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    container_client = blob_service_client.get_container_client(os.getenv('CONTAINER_NAME'))
    blob_client = container_client.get_blob_client(file_name)

    json_data = json.dumps(dictionary)
    blob_client.upload_blob(json_data, overwrite=True)

def load_csv_np(filepath, skip_header):
    return np.genfromtxt(filepath, delimiter=";", skip_header=skip_header)

def save_csv(filename, rows):
    """Save a list of rows to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(rows)

def try_float(value):
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None

#endregion