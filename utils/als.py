import os
from collections import defaultdict
import pandas as pd
import implicit
from implicit.recommender_base import RecommenderBase
import numpy as np
import implicit
from scipy.sparse import csr_matrix

from utils.classes.Settings import Settings
from utils.classes.StringBuilder import StringBuilder
from DTO.ProductDTO import ProductDTO
from utils import shared
from utils.emailing import Emailing

ID_COL_NAME = 'id'
STOCK_COL_NAME = 'stock'
STATUS_COL_NAME = 'status'
VISIBILITY_COL_NAME = 'visibility'
ACTIVE_COL_NAME = 'active'
TITLE_COL_NAME = 'title'
CATEGORIES_COL_NAME = 'categories'
MANUFACTURER_COL_NAME = 'manufacturer'
PRICE_COL_NAME = 'price'

# FT: We can not pass partial interactions because of timestamp updates
def get_recommendation_result(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    sparse_user_product_matrix, products, user_ids = shared.save_interaction_values(raw_interactions, raw_products)

    model = train_model(sparse_user_product_matrix)

    return get_recommendation_result_dict(model, sparse_user_product_matrix, user_ids, products)

def train_model(sparse_user_product):
    now = pd.Timestamp.now()
    sb = StringBuilder()

    model = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, alpha=1.0, iterations=15, calculate_training_loss=True) # FT: calculate_training_loss needs to be true if we want to fit_callback work
    model.fit_callback = store_loss(sb)
    model.fit(sparse_user_product, show_progress=False)

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Model training", sb.__str__())

    return model

def get_recommendation_result_dict(model: RecommenderBase, sparse_user_product_matrix: csr_matrix, user_ids: pd.Index, products: pd.DataFrame) -> dict:
    now = pd.Timestamp.now()
    sb = StringBuilder()

    product_indexes_to_filter = get_product_indexes_to_filter(products)
    sb.append(f'Products to filter count: {len(product_indexes_to_filter)}\n')

    recommendations_dict = defaultdict(list)
    
    recommendations_dict['top_ten_overall_recommendations'] = get_top_overall_recommendations(sparse_user_product_matrix, products, product_indexes_to_filter)
    sb.append(f"Top ten overall recommendations: {get_products_for_display(recommendations_dict['top_ten_overall_recommendations'])}\n")

    batch_size = 1000
    to_generate = np.arange(len(user_ids))
    
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx : startidx + batch_size]
        product_indexes, scores = model.recommend(batch, sparse_user_product_matrix[batch], filter_already_liked_items=False, filter_items=product_indexes_to_filter)
        for i, user_index in enumerate(batch):
            user_id = str(user_ids[user_index])
            products_for_recommendation = []
            for product_index, score in zip(product_indexes[i], scores[i]):
                product = products.iloc[product_index]
                productDTO = init_productDTO(product)
                products_for_recommendation.append(productDTO.__dict__)
            recommendations_dict[user_id] = products_for_recommendation
    
    sb.append(f"Top ten '{Settings().EMAIL_RECOMMENDATIONS_TEST}' recommendations: {get_products_for_display(recommendations_dict[Settings().EMAIL_RECOMMENDATIONS_TEST])}\n")

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Making recommendations", sb.__str__())

    return recommendations_dict

def get_product_indexes_to_filter(products: pd.DataFrame) -> pd.Index:
    return products.loc[
        (products[STOCK_COL_NAME] == 0) |
        (products[STATUS_COL_NAME] != 'Published') |
        (products[VISIBILITY_COL_NAME] != 'Public') |
        (products[ACTIVE_COL_NAME] != True)
    ].index

def get_top_overall_recommendations(sparse_user_product_matrix: csr_matrix, products: pd.DataFrame, product_indexes_to_filter: pd.Index) -> list[dict]:
    result: list[dict] = []

    # FT: Because we are not modifying product_ratings ravel is faster then flatten
    product_ratings = np.array(sparse_user_product_matrix.sum(axis=0)).ravel()

    rated_product_counts = sparse_user_product_matrix.getnnz(axis=0)

    avg_interactions = np.where(
        rated_product_counts == 0,
        0.0,
        product_ratings / rated_product_counts
    )

    valid_products = products.drop(product_indexes_to_filter).reset_index(drop=True)
    valid_avg_interactions = np.delete(avg_interactions, product_indexes_to_filter)

    top_10_valid_indices = np.argsort(valid_avg_interactions)[-10:][::-1]
    top_10_products = valid_products.iloc[top_10_valid_indices]

    for _, product in top_10_products.iterrows():
        productDTO = init_productDTO(product)
        result.append(productDTO.__dict__)

    return result

def get_products_for_display(products: list[dict]) -> str:
    product_ids = [str(product['Id']) for product in products]
    return ', '.join(product_ids)

def init_productDTO(row: pd.Series):
    return ProductDTO(
        Id=row[ID_COL_NAME], 
        SKU=row[ID_COL_NAME],
        Stock=row[STOCK_COL_NAME], 
        Status=row[STATUS_COL_NAME], 
        Visibility=row[VISIBILITY_COL_NAME], 
        Active=row[ACTIVE_COL_NAME],
        Title=row[TITLE_COL_NAME],
        Categories=row[CATEGORIES_COL_NAME],
        Manufacturer=row[MANUFACTURER_COL_NAME],
        Price=row[PRICE_COL_NAME],
    )

# https://github.com/benfred/implicit/issues/281
def store_loss(output: StringBuilder): 
    def inner(iteration, elapsed, loss): 
        output.append(f'Loss {loss:.5f}\n') 

    return inner