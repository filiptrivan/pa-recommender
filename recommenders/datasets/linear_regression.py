import os
import csv
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import date

USER_COL_NAME = 'UserId'
PRODUCT_COL_NAME = 'ProductId'
RATING_COL_NAME = 'Rating'
BOUGHT_COL_NAME = 'Bought'
PUT_IN_CART_COL_NAME = 'PutInCart'
PUT_IN_FAVORITE_COL_NAME = 'PutInFavorite'
CLICKED_COL_NAME = 'Clicked'

def save_rating_values():
    ratings = load_csv_list("../../recommenders/datasets/pa/ratings.csv")

    pivot_data = defaultdict(dict)

    for row in ratings[1:]:
        userId = row[USER_COL_NAME]
        productId = row[PRODUCT_COL_NAME]

        if row[RATING_COL_NAME]:
            rating = try_float(row[RATING_COL_NAME])
        elif int(row[BOUGHT_COL_NAME]) != 0:
            rating = 5.0
        elif int(row[PUT_IN_CART_COL_NAME]) != 0:
            rating = 4.5
        elif int(row[PUT_IN_FAVORITE_COL_NAME]) != 0:
            rating = 4.0
        elif int(row[CLICKED_COL_NAME]) != 0:
            rating = 3.5
        else:
            rating = ''

        product_seasons = get_product_seasons(productId)
        current_season = get_current_season()

        if current_season in product_seasons:
            if rating == '':
                rating = 0
            rating += 1
        
        pivot_data[productId][userId] = rating

    userIds = sorted({ row[USER_COL_NAME] for row in ratings[1:] })
    productIds = sorted(pivot_data.keys())

    new_csv = []

    for productId in productIds:
        row = []
        for userId in userIds:
            row.append(pivot_data[productId].get(userId, ''))
        new_csv.append(row)

    save_csv('ratings_mean.csv', new_csv)

def get_data():
    X = load_csv_np("../../recommenders/datasets/pa/product_features.csv", skip_header=True) # features for products (Bosch, Makita, DeWalt, Burgija, Testera...) 4 X 10000
    num_features = X.shape[1]
    num_products = X.shape[0]

    Y_with_nan = load_csv_np("ratings_mean.csv", skip_header=False) # 4 X 3
    num_users = Y_with_nan.shape[1]

    R = get_rated_notrated_matrix(Y_with_nan)

    Y = np.nan_to_num(Y_with_nan, nan=0)

    print("Y", Y.shape, "R", R.shape)
    print("X", X.shape)
    print("num_features", num_features)
    print("num_products",   num_products)
    print("num_users",    num_users)

    return X, Y, R, num_features, num_products

def get_rated_notrated_matrix(Y_with_nan):
    return np.where(np.isnan(Y_with_nan), 0, 1)

def normalize_ratings(Y, R):
    """
    Preprocess data by subtracting mean rating for every row.
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalize_ratings(Y, R) normalized Y so that each row
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1) 
    Ynorm = Y - np.multiply(Ymean, R)
    return(Ynorm, Ymean)

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_products,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_products,num_users)    : matrix of user ratings of products
      R (ndarray (num_products,num_users)    : matrix, where R(i, j) = 1 if the i-th products was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

#region Helpers

def load_csv_list(filepath):
    """Load CSV data from a file and return a list of rows."""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]

def load_csv_np(filepath, skip_header):
    return np.genfromtxt(filepath, delimiter=",", skip_header=skip_header)

def save_csv(filename, rows):
    """Save a list of rows to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def try_float(value):
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None

def get_product_seasons(productId):
    features_for_products = load_csv_list("../../recommenders/datasets/pa/product_features.csv") # features for products (Bosch, Makita, DeWalt, Burgija, Testera...) 4 X 10000

    result = []

    for product in features_for_products:
        if product["ProductId"] == str(productId):
            if product["Summer"] == "1":
                result.append("Summer")
            if product["Autumn"] == "1":
                result.append("Autumn")
            if product["Winter"] == "1":
                result.append("Winter")
            if product["Spring"] == "1":
                result.append("Spring")
    
    return result

def get_current_season():
    now = date.today()
    month = now.month
    day = now.day

    if (month == 12 and day >= 21) or (1 <= month <= 3 and not (month == 3 and day >= 21)):
        return 'Winter'
    elif (month == 3 and day >= 21) or (4 <= month <= 6 and not (month == 6 and day >= 21)):
        return 'Spring'
    elif (month == 6 and day >= 21) or (7 <= month <= 9 and not (month == 9 and day >= 23)):
        return 'Summer'
    else:
        return 'Autumn'

#endregion