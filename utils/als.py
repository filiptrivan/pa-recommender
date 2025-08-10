import os
from collections import defaultdict
import pandas as pd
import implicit
from implicit.recommender_base import RecommenderBase
import numpy as np
import implicit
from scipy.sparse import csr_matrix
import json

from utils.classes.Settings import Settings
from utils.classes.StringBuilder import StringBuilder
from DTO.ProductDTO import ProductDTO
from utils import shared
from utils.emailing import Emailing
import redis

ID_COL_NAME = 'id'
STOCK_COL_NAME = 'stock'
STATUS_COL_NAME = 'status'
TITLE_COL_NAME = 'title'

#region Homepage And Similar Products Recommender

HOMEPAGE_RECOMMENDER_REDIS = redis.Redis(
    host='redis-13503.c293.eu-central-1-1.ec2.redns.redis-cloud.com',
    port=13503,
    decode_responses=True,
    username=Settings().REDIS_USERNAME,
    password=Settings().REDIS_PASSWORD
)

TOP_TEN_OVERALL_RECOMMENDATIONS_KEY = 'top_ten_overall_recommendations'
HOMEPAGE_RECOMMENDER_REDIS_KEY_EXPIRATION = 604800 # 7 days

SIMILAR_PRODUCTS_RECOMMENDER_REDIS = redis.Redis(
    host='redis-18870.crce198.eu-central-1-3.ec2.redns.redis-cloud.com',
    port=18870,
    decode_responses=True,
    username=Settings().REDIS_USERNAME,
    password=Settings().REDIS_PASSWORD
)

SIMILAR_PRODUCTS_RECOMMENDER_REDIS_KEY_EXPIRATION = 604800 # 7 days


# We can not pass partial interactions because of timestamp updates
def process_homepage_and_similar_products_recommendations(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    sparse_user_product_matrix, user_ids, products = shared.get_homepage_and_similar_products_interaction_values(raw_interactions, raw_products)

    model = homepage_and_similar_products_train_model(sparse_user_product_matrix)

    save_homepage_and_similar_products_recommendations(model, sparse_user_product_matrix, user_ids, products)

def homepage_and_similar_products_train_model(sparse_user_product):
    now = pd.Timestamp.now()
    sb = StringBuilder()

    # calculate_training_loss needs to be true if we want to fit_callback work
    model = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, alpha=1.0, iterations=15, calculate_training_loss=True) 
    model.fit_callback = store_loss(sb)
    model.fit(sparse_user_product, show_progress=False)

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Homepage and similar products model training", sb.__str__())

    return model

def save_homepage_and_similar_products_recommendations(
    model: RecommenderBase, 
    sparse_user_product_matrix: csr_matrix, 
    user_ids: pd.Index, 
    products: pd.DataFrame
):
    now = pd.Timestamp.now()
    processingLog = StringBuilder()

    product_indexes_to_filter = get_product_indexes_to_filter(products)
    processingLog.append(f'Products to filter count: {len(product_indexes_to_filter)}\n')

    save_homepage_recommendations(model, sparse_user_product_matrix, user_ids, products, product_indexes_to_filter, processingLog)
    save_similar_products_recommendations(model, products, product_indexes_to_filter, processingLog)
    
    processingLog.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Saving homepage and similar products recommendations", processingLog.__str__())

def save_homepage_recommendations(
    model: RecommenderBase, 
    sparse_user_product_matrix: csr_matrix, 
    user_ids: pd.Index, 
    products: pd.DataFrame, 
    product_indexes_to_filter: pd.Index,
    processingLog: StringBuilder,
) -> dict:
    processingLog.append('\nHomepage recommendations saving\n')

    recommendations_dict = defaultdict(list)

    recommendations_dict[TOP_TEN_OVERALL_RECOMMENDATIONS_KEY] = get_top_overall_recommendations(sparse_user_product_matrix, products, product_indexes_to_filter)
    processingLog.append(f"Top ten overall recommendations: {get_products_for_display(recommendations_dict[TOP_TEN_OVERALL_RECOMMENDATIONS_KEY])}\n")

    batch_size = 1000
    to_generate = np.arange(len(user_ids))

    redis_pipeline = HOMEPAGE_RECOMMENDER_REDIS.pipeline()

    try:
        redis_pipeline.set(name=TOP_TEN_OVERALL_RECOMMENDATIONS_KEY, ex=HOMEPAGE_RECOMMENDER_REDIS_KEY_EXPIRATION, value=json.dumps(recommendations_dict[TOP_TEN_OVERALL_RECOMMENDATIONS_KEY])) # ex=7 days
        for startidx in range(0, len(to_generate), batch_size):
            batch = to_generate[startidx : startidx + batch_size]
            product_indexes, _ = model.recommend(batch, sparse_user_product_matrix[batch], filter_already_liked_items=False, filter_items=product_indexes_to_filter)
            # TODO: Check if, after the real training, there are products which should be filtered
            for i, user_index in enumerate(batch):
                user_id = user_ids[user_index] # Not casting here improved performance for 10 sec for 500k interactions
                products_for_recommendation = []
                for product_index in product_indexes[i]:
                    product = products.iloc[product_index]
                    productDTO = init_productDTO(product)
                    products_for_recommendation.append(productDTO.Id)
                recommendations_dict[user_id] = products_for_recommendation
                redis_pipeline.set(name=user_id, ex=HOMEPAGE_RECOMMENDER_REDIS_KEY_EXPIRATION, value=json.dumps(products_for_recommendation)) # ex=7 days
        redis_pipeline.execute()
    except Exception as ex:
        redis_pipeline.reset()
        raise ex
        
    test_email_for_recommendations = Settings().TEST_EMAIL_FOR_RECOMMENDATIONS

    if len(recommendations_dict[test_email_for_recommendations]) == 0:
        test_email_for_recommendations = user_ids[0]
    
    test_recommendations_for_display = get_products_for_display(recommendations_dict[test_email_for_recommendations])

    processingLog.append(f"Top ten '{test_email_for_recommendations}' recommendations: {test_recommendations_for_display}\n")

    return recommendations_dict

def get_top_overall_recommendations(sparse_user_product_matrix: csr_matrix, products: pd.DataFrame, product_indexes_to_filter: pd.Index) -> list[dict]:
    result: list[int] = []

    # Because we are not modifying product_ratings ravel is faster then flatten
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
        result.append(productDTO.Id)

    return result

def save_similar_products_recommendations(
    model: RecommenderBase, 
    products: pd.DataFrame,
    product_indexes_to_filter: pd.Index,
    processingLog: StringBuilder
):
    processingLog.append('\nSimilar products recommendations saving\n')
    
    product_ids = products[ID_COL_NAME].tolist() # Only tolist() worked to convert from int32 to int

    recommendations_dict = defaultdict(list)

    batch_size = 1000
    to_generate = np.arange(len(product_ids))

    redis_pipeline = SIMILAR_PRODUCTS_RECOMMENDER_REDIS.pipeline()

    try:
        for startidx in range(0, len(to_generate), batch_size):
            batch = to_generate[startidx : startidx + batch_size]
            product_for_recommendation_indexes, _ = model.similar_items(batch, filter_items=product_indexes_to_filter)
            for i, product_to_recommend_index in enumerate(batch):
                product_to_recommend_id = product_ids[product_to_recommend_index]
                similar_products = []
                for product_index in product_for_recommendation_indexes[i]:
                    product_id = product_ids[product_index]
                    if product_id != product_to_recommend_id: # Skip itself, we don't want to show itself
                        similar_products.append(product_id)
                recommendations_dict[product_to_recommend_id] = similar_products
                redis_pipeline.set(name=product_to_recommend_id, ex=SIMILAR_PRODUCTS_RECOMMENDER_REDIS_KEY_EXPIRATION, value=json.dumps(similar_products)) # ex=7 days
        redis_pipeline.execute()
    except Exception as ex:
        redis_pipeline.reset()
        raise ex

    test_product_for_similar_products = Settings().TEST_PRODUCT_FOR_SIMILAR_PRODUCTS

    if len(recommendations_dict[test_product_for_similar_products]) == 0:
        test_product_for_similar_products = product_ids[0]
    
    test_similar_products_for_display = ', '.join([str(product_id) for product_id in recommendations_dict[test_product_for_similar_products]])

    processingLog.append(f"Top ten '{test_product_for_similar_products}' similar products: {test_similar_products_for_display}\n")

    return recommendations_dict

#endregion

#region Cross Sell Recommender

CROSS_SELL_RECOMMENDER_REDIS = redis.Redis(
    host='redis-19259.c135.eu-central-1-1.ec2.redns.redis-cloud.com',
    port=19259,
    decode_responses=True,
    username=Settings().REDIS_USERNAME,
    password=Settings().REDIS_PASSWORD
)

# We can not pass partial interactions because of timestamp updates
def process_cross_sell_recommendation(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    sparse_product_product_matrix, product_to_recommend_ids, products_for_recommendation = shared.get_cross_sell_interaction_values(raw_interactions, raw_products)

    model = cross_sell_train_model(sparse_product_product_matrix)

    save_cross_sell_recommendations(model, sparse_product_product_matrix, product_to_recommend_ids, products_for_recommendation)

def cross_sell_train_model(sparse_user_product):
    now = pd.Timestamp.now()
    sb = StringBuilder()

    model = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, alpha=1.0, iterations=15, calculate_training_loss=True) # calculate_training_loss needs to be true if we want to fit_callback work
    model.fit_callback = store_loss(sb)
    model.fit(sparse_user_product, show_progress=False)

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Cross sell model training", sb.__str__())

    return model

def save_cross_sell_recommendations(model: RecommenderBase, sparse_product_product_matrix: csr_matrix, product_to_recommend_ids: pd.Index, products_for_recommendation: pd.DataFrame) -> dict:
    now = pd.Timestamp.now()
    sb = StringBuilder()

    product_indexes_to_filter = get_product_indexes_to_filter(products_for_recommendation)
    sb.append(f'Products to filter count: {len(product_indexes_to_filter)}\n')

    recommendations_dict = defaultdict(list)

    batch_size = 1000
    to_generate = np.arange(len(product_to_recommend_ids))
    
    redis_pipeline = CROSS_SELL_RECOMMENDER_REDIS.pipeline()

    try:
        for startidx in range(0, len(to_generate), batch_size):
            batch = to_generate[startidx : startidx + batch_size]
            product_for_recommendation_indexes, _ = model.recommend(batch, sparse_product_product_matrix[batch], filter_already_liked_items=False, filter_items=product_indexes_to_filter)
            for i, product_to_recommend_index in enumerate(batch):
                product_to_recommend_id = product_to_recommend_ids[product_to_recommend_index]
                products_for_cross_sell = []
                for product_index in product_for_recommendation_indexes[i]:
                    product_id = products_for_recommendation.iloc[product_index][ID_COL_NAME]
                    if product_id != product_to_recommend_id: # Skip itself, we don't want to show itself
                        products_for_cross_sell.append(product_id)
                recommendations_dict[product_to_recommend_id] = products_for_cross_sell
                # Persistant, because it's product to product interactions, for this we need the best data possible, we shouldn't retrain it in short period
                redis_pipeline.set(name=product_to_recommend_id, value=json.dumps(products_for_cross_sell))
        redis_pipeline.execute()
    except Exception as ex:
        redis_pipeline.reset()
        raise ex
    
    test_product_for_cross_sell = Settings().TEST_PRODUCT_FOR_CROSS_SELL

    if len(recommendations_dict[test_product_for_cross_sell]) == 0:
        test_product_for_cross_sell = product_to_recommend_ids[0]
    
    test_recommendations_for_display = ', '.join(recommendations_dict[test_product_for_cross_sell])

    sb.append(f"Top ten '{test_product_for_cross_sell}' recommendations: {test_recommendations_for_display}\n")

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Storing recommendations for cross sell", sb.__str__())

    return recommendations_dict

#endregion

#region Shared

def get_products_for_display(product_ids: list[int]) -> str:
    product_ids = [str(product_id) for product_id in product_ids]
    return ', '.join(product_ids)

def init_productDTO(row: pd.Series):
    return ProductDTO(
        Id=row[ID_COL_NAME], 
        # SKU=row[ID_COL_NAME],
        Stock=row[STOCK_COL_NAME], 
        Status=row[STATUS_COL_NAME], 
        Title=row[TITLE_COL_NAME]
    )

# https://github.com/benfred/implicit/issues/281
def store_loss(output: StringBuilder): 
    def inner(iteration, elapsed, loss): 
        output.append(f'Loss {loss:.5f}\n') 

    return inner

def get_product_indexes_to_filter(products: pd.DataFrame) -> pd.Index:
    return products.loc[
        (products[STOCK_COL_NAME] == 0) |
        (products[STATUS_COL_NAME] != 'Published')
    ].index

#endregion