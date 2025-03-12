import csv
import pandas as pd
from collections import defaultdict
import numpy as np
from math import exp
from math import log1p

#region Data Manipulation

PRODUCT_COL_NAME = 'ProductId'
USER_COL_NAME = 'UserId'
INTERACTION_COL_NAME = 'Interaction'

RECENCY_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.
MULTIPLE_INTERACTIONS_DECAY_SCALE = 20 # FT: Adjust to fine-tune how quickly the weight drops. A smaller value will lead to a very steep drop, while a larger value will make the decay more gradual.

def save_interaction_values(interactions_path, all_products):
    now = pd.Timestamp.now()
    interactions = load_excel_list(interactions_path)

    pivot_data = defaultdict(dict)

    grouped_ratings = defaultdict(list)

    for row in interactions:
        key = (str(row[PRODUCT_COL_NAME]), row[USER_COL_NAME])
        grouped_ratings[key].append(row)

    for (productId, userId), rows in grouped_ratings.items():
        rating = 0
        
        for i in range(len(rows)):
            row = rows[i]

            # FT: Recency bonus
            timestamp = parse_and_format_timestamp(row['Timestamp'])
            # helper = row['Timestamp']
            # print(f'{helper} -> {timestamp}')
            diff_days  = (now - timestamp).total_seconds() / (60 * 60 * 24)

            if diff_days < 0:
                raise ValueError("The timestamp is in the future; please provide a valid past timestamp.")

            if row[INTERACTION_COL_NAME] == 'Bought':
                rating += get_rating_based_on_recency(diff_days, 1)
            elif row[INTERACTION_COL_NAME] == 'PutInCart':
                rating += get_rating_based_on_recency(diff_days, 0.5)
            elif row[INTERACTION_COL_NAME] == 'PutInFavorites':
                rating += get_rating_based_on_recency(diff_days, 0.3)
            elif row[INTERACTION_COL_NAME] == 'Clicked':
                rating += get_rating_based_on_recency(diff_days, 0.1)
            else:
                raise ValueError("Interaction value doesn't exist.")

        rating += get_multiple_interaction_bonus(len(rows), rating)

        pivot_data[productId][userId] = '' if rating == 0  else rating

    productIds = sorted(pivot_data.keys())
    userIds = sorted({user[USER_COL_NAME] for user in interactions})
    
    products = []
    users = []
    new_csv = []

    for user_id in userIds:
        users.append([user_id])

    for productId in productIds:
        row = []
        for userId in userIds:
            row.append(pivot_data[productId].get(userId, ''))
        product = next((x for x in all_products if x['SKU'] == productId), None)
        products.append([
            productId,
            product.get('Stock', '0') if product else '0',
            product.get('Status', 'Draft') if product else 'Draft',
            product.get('Visibility', 'Private') if product else 'Private',
            int(product.get('Active', '0')) if product else '0'
        ])
        new_csv.append(row)

    save_csv('Interactions.csv', new_csv)
    save_csv('Products.csv', products)
    save_csv('Users.csv', users)

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
        print(f"Day {day}: {value:.4f}")

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

def get_data():
    Y_with_nan = load_csv_np("Interactions.csv", skip_header=False)
    num_users = Y_with_nan.shape[1]
    num_products = Y_with_nan.shape[0]

    Y = np.nan_to_num(Y_with_nan, nan=0)

    print("Y", Y.shape)
    print("num_products",   num_products)
    print("num_users",    num_users)

    return Y

#endregion

#region Helpers

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