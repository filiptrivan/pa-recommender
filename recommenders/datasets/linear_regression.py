import os
import csv
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
from math import exp

#region Data Manipulation

USER_COL_NAME = 'UserId'
PRODUCT_COL_NAME = 'ProductId'
INTERACTION_COL_NAME = 'Bought'

DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.

def save_rating_values():
    ratings = load_csv_list("../../recommenders/datasets/pa/ratings.csv")

    pivot_data = defaultdict(dict)

    grouped_ratings = defaultdict(list)

    for row in ratings[1:]:
        key = (row[USER_COL_NAME], row[PRODUCT_COL_NAME])
        grouped_ratings[key].append(row)

    for (userId, productId), rows in grouped_ratings.items():
        rating = 0
        
        bc = 1
        picc = 1
        pifc = 1
        cc = 1
        for row in rows:
            if row[INTERACTION_COL_NAME] == 'Bought':
                rating += 4.0 / bc
            elif row[INTERACTION_COL_NAME] == 'PutInCart':
                rating += 2.0 / picc
            elif row[INTERACTION_COL_NAME] == 'PutInFavorites':
                rating += 1.5 / pifc
            elif row[INTERACTION_COL_NAME] == 'Clicked':
                rating += 0.5 / cc

            # FT: Recency bonus
            if row['Timestamp']:
                timestamp = date(row['Timestamp'])
                now = datetime.now()
                diff_days = (now - timestamp).total_seconds() / (60 * 60 * 24)
                bonus = recency_bonus(diff_days)
                rating += bonus

        # TODO FT: Put this logic after everything to filter products
        # product_seasons = get_product_seasons(productId)
        # current_season = get_current_season()

        # if current_season in product_seasons:
        #     rating += 1
        
        pivot_data[productId][userId] = '' if rating == 0  else rating

    userIds = sorted({ row[USER_COL_NAME] for row in ratings[1:] })
    productIds = sorted(pivot_data.keys())

    new_csv = []

    for productId in productIds:
        row = []
        for userId in userIds:
            row.append(pivot_data[productId].get(userId, ''))
        new_csv.append(row)

    save_csv('ratings_mean.csv', new_csv)

def recency_bonus(diff_days, decay_constant):
    """
    Returns a recency bonus that decays rapidly initially but approaches 0.5 for old interactions.
    At diff_days=0, bonus=1; as diff_days -> infinity, bonus -> 0.5.
    """
    return 0.5 + 0.5 * exp(-diff_days / decay_constant)

def test_recency_bonus(days, decay_scale=DECAY_SCALE):
    first_days = {day: 2 * exp(-day / decay_scale) for day in range(1, days)}

    for day, value in first_days.items():
        print(f"Day {day}: {value:.4f}")

def get_data():
    X = load_csv_np("../../recommenders/datasets/pa/product_features.csv", skip_header=True) # features for products (Bosch, Makita, DeWalt, Burgija, Testera...) 4 X 10000
    X = X[:, 1:]
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

#endregion

#region Algorithm

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

def rmse(Y, Y_predictions, R):
    mask = R == 1
    rmse_score = np.sqrt(np.mean((Y[mask] - Y_predictions[mask]) ** 2))
    print(f'RMSE score: {rmse_score}')
    return rmse_score

#region TensorFlow

def initialize_tf_variables(product_features, num_features, num_users):
    tf.random.set_seed(1234)
    X = tf.Variable(product_features, dtype=tf.float64, name='X', trainable=True)
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

    return X, W, b

def calculate_parameters(X, W, b, Ynorm, R, iterations, lambda_, learning_rate):
    optimizer = keras.optimizers.Adam(learning_rate)

    cost_history = []

    for iter in range(iterations):
        # Use TensorFlow’s GradientTape to record the operations used to compute the cost 
        with tf.GradientTape() as tape:
            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss
        grads = tape.gradient( cost_value, [X,W,b] )

        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip(grads, [X,W,b]) )

        cost_history.append(cost_value)

        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.3f}")

    plt.plot(range(iterations), cost_history, label="Cost Function")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Over Iterations")
    plt.legend()
    plt.show()

#endregion

#endregion


#region Helpers

def load_csv_list(filepath):
    """Load CSV data from a file and return a list of rows."""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return [row for row in reader]

def load_csv_np(filepath, skip_header):
    return np.genfromtxt(filepath, delimiter=";", skip_header=skip_header)

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