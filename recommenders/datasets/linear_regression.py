import os
import csv
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras

USER_COL_NAME = 'UserId'
PRODUCT_COL_NAME = 'ProductId'
RATING_COL_NAME = 'Rating'
BOUGHT_COL_NAME = 'Bought'
PUT_IN_CART_COL_NAME = 'PutInCart'
PUT_IN_FAVORITE_COL_NAME = 'PutInFavorite'
CLICKED_COL_NAME = 'Clicked'

def save_rating_values():
    ratings = load_csv_list("../../recommenders/datasets/pa/ratings.csv")

    header = ratings[0]
    USER_COL = header.index(USER_COL_NAME)
    PRODUCT_COL = header.index(PRODUCT_COL_NAME)
    RATING_COL = header.index(RATING_COL_NAME)
    BOUGHT_COL = header.index(BOUGHT_COL_NAME)
    PUT_IN_CART_COL = header.index(PUT_IN_CART_COL_NAME)
    PUT_IN_FAVORITE_COL = header.index(PUT_IN_FAVORITE_COL_NAME)
    CLICKED_COL = header.index(CLICKED_COL_NAME)

    pivot_data = defaultdict(dict)

    for row in ratings[1:]:
        userId = row[USER_COL]
        productId = row[PRODUCT_COL]

        if row[RATING_COL]:
            rating = row[RATING_COL]
        elif int(row[BOUGHT_COL]) != 0:
            rating = '5'
        elif int(row[PUT_IN_CART_COL]) != 0:
            rating = '4.5'
        elif int(row[PUT_IN_FAVORITE_COL]) != 0:
            rating = '4'
        elif int(row[CLICKED_COL]) != 0:
            rating = '3.5'
        else:
            rating = ''
        
        pivot_data[productId][userId] = rating

    users = sorted({ row[USER_COL] for row in ratings[1:] })
    products = sorted(pivot_data.keys())

    new_csv = []

    for product in products:
        row = []
        for user in users:
            row.append(pivot_data[product].get(user, ''))
        new_csv.append(row)

    save_csv('ratings_mean.csv', new_csv)

def get_data():
    X = load_csv_np("../../recommenders/datasets/pa/product_features.csv", skip_header=True) # features for products (Bosch, Makita, DeWalt, Burgija, Testera...) 4 X 10000
    num_features = X.shape[1]
    num_products = X.shape[0]

    Y = load_csv_np("ratings_mean.csv", skip_header=False) # 4 X 3
    num_users = Y.shape[1]

    R = get_rated_notrated_matrix()

    print("Y", Y.shape, "R", R.shape)
    print("X", X.shape)
    print("num_features", num_features)
    print("num_products",   num_products)
    print("num_users",    num_users)

    return X, Y, R, num_features, num_products

def get_rated_notrated_matrix():
    Y = load_csv_np("ratings_mean.csv", skip_header=False)
    return np.where(np.isnan(Y), 0, 1)
    return np.sum((np.dot(W, X) + b) - Y)

def normalize_ratings(Y, R):
    """
    Normalizes the ratings in Y so that each product has a zero mean.
    
    Parameters:
        Y (np.array): A (num_products x num_users) matrix of ratings.
        R (np.array): A (num_products x num_users) binary indicator matrix where R[i, j] = 1 
                      if product i was rated by user j, and 0 otherwise.
    
    Returns:
        Ynorm (np.array): The normalized ratings matrix with the mean subtracted for each product.
        Ymean (np.array): A vector of mean ratings for each product.
    """
    num_products = Y.shape[0]
    Ymean = np.zeros(num_products)
    Ynorm = np.zeros(Y.shape)
    
    # Iterate over each product
    for i in range(num_products):
        # Find the indices of users who rated the product
        idx = np.where(R[i, :] == 1)[0]
        
        if idx.size > 0:
            # Compute the mean rating for the product
            Ymean[i] = np.mean(Y[i, idx])
            # Subtract the mean rating from each rating where a rating exists
            Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    
    return Ynorm, Ymean.reshape(-1, 1)

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
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)

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
    
#endregion