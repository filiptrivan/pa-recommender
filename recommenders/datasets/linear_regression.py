import os
import csv

USER_COL_NAME = 'UserId'
PRODUCT_COL_NAME = 'ProductId'
RATING_COL_NAME = 'Rating'
BOUGHT_COL_NAME = 'Bought'
PUT_IN_CART_COL_NAME = 'PutInCart'
PUT_IN_FAVORITE_COL_NAME = 'PutInFavorite'
CLICKED_COL_NAME = 'Clicked'

def save_rating_values():
    csv = load_csv("../../recommenders/datasets/pa/ratings.csv")
    new_csv = []
    new_csv.append(["UserId", "ProductId", "Rating"])

    header = csv[0]
    USER_COL = header.index(USER_COL_NAME)
    PRODUCT_COL = header.index(PRODUCT_COL_NAME)
    RATING_COL = header.index(RATING_COL_NAME)
    BOUGHT_COL = header.index(BOUGHT_COL_NAME)
    PUT_IN_CART_COL = header.index(PUT_IN_CART_COL_NAME)
    PUT_IN_FAVORITE_COL = header.index(PUT_IN_FAVORITE_COL_NAME)
    CLICKED_COL = header.index(CLICKED_COL_NAME)


    for row in csv[1:]:
        if try_float(row[RATING_COL]):
            new_csv.append([row[USER_COL], row[PRODUCT_COL], row[RATING_COL]])

        elif int(row[BOUGHT_COL]) != 0:
            new_csv.append([row[USER_COL], row[PRODUCT_COL], 5])

        elif int(row[PUT_IN_CART_COL]) != 0:
            new_csv.append([row[USER_COL], row[PRODUCT_COL], 4.5])

        elif int(row[PUT_IN_FAVORITE_COL]) != 0:
            new_csv.append([row[USER_COL], row[PRODUCT_COL], 4])

        elif int(row[CLICKED_COL]) != 0:
            new_csv.append([row[USER_COL], row[PRODUCT_COL], 3.5])

        else:
            new_csv.append([row[USER_COL], row[PRODUCT_COL], None])

    save_csv('ratings_mean.csv', new_csv)

def load_csv(filename):
    """Load CSV data from a file and return a list of rows."""
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)

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