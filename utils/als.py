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
    host='redis-18870.crce198.eu-central-1-3.ec2.redns.redis-cloud.com',
    port=18870,
    decode_responses=True,
    username=Settings().REDIS_USERNAME,
    password=Settings().REDIS_PASSWORD
)

TOP_TEN_OVERALL_RECOMMENDATIONS_KEY = 'top_ten_overall_recommendations'
HOMEPAGE_RECOMMENDER_REDIS_KEY_EXPIRATION = 604800 # 7 days

# We can not pass partial interactions because of timestamp updates
def process_homepage_and_similar_products_recommendations(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame):
    sparse_user_product_matrix, user_ids, products = shared.get_homepage_and_similar_products_interaction_values(raw_interactions, raw_products)

    model = homepage_and_similar_products_train_model(sparse_user_product_matrix)

    save_homepage_and_similar_products_recommendations(model, sparse_user_product_matrix, user_ids, products)

def homepage_and_similar_products_train_model(sparse_user_product):
    now = pd.Timestamp.now()
    sb = StringBuilder()

    sb.append(f'(number of users, number of products): {sparse_user_product.shape}\n')
    sb.append(f'Number of non-zero interactions: {sparse_user_product.nnz}\n')
    sb.append(f'Density: {sparse_user_product.nnz / (sparse_user_product.shape[0] * sparse_user_product.shape[1]):.8f}\n\n')

    percentiles = np.percentile(sparse_user_product.data, [0, 25, 50, 75, 100])
    labels = ["0%", "25%", "50%", "75%", "100%"]
    for l, p in zip(labels, percentiles):
        sb.append(f"{l}: {p:.4f}\n")

    # α controls how much more important observed interactions are compared to unobserved ones (which always have weight = 1).
    # Too small → model ignores your positives, loss is tiny, recommendations are generic.
    # Too large → model overfits to observed interactions, ignores zeros, may recommend only very popular items.

    # regularization penalize too high interactions, we don't need it too high because in the data preparation we thought about this

    # calculate_training_loss needs to be true if we want to fit_callback work
    # model = implicit.als.AlternatingLeastSquares(factors=550, regularization=0.01, alpha=140.0, iterations=15, calculate_training_loss=True) 
    model = implicit.als.AlternatingLeastSquares(factors=550, regularization=0.01, alpha=1.0, iterations=15, calculate_training_loss=True) 
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
    # processingLog.append(f"Product ids to filter: {', '.join(map(str, product_indexes_to_filter))}\n")
    processingLog.append(f'Products to filter count: {len(product_indexes_to_filter)}\n')

    save_homepage_recommendations(model, sparse_user_product_matrix, user_ids, products, product_indexes_to_filter, processingLog)
    
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

    recommendations_dict[TOP_TEN_OVERALL_RECOMMENDATIONS_KEY], titles = get_top_overall_recommendations(sparse_user_product_matrix, products, product_indexes_to_filter)
    processingLog.append(f"Top ten overall recommendations: {get_products_for_display(titles)}\n")

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

def get_top_overall_recommendations(sparse_user_product_matrix: csr_matrix, products: pd.DataFrame, product_indexes_to_filter: list[int]) -> tuple[list[int], list[str]]:
    # Because we are not modifying product_ratings ravel is faster then flatten
    product_ratings = np.asarray(sparse_user_product_matrix.sum(axis=0)).ravel()

    # Drop by position
    mask = np.ones(len(products), dtype=bool)
    mask[product_indexes_to_filter] = False

    valid_products = products[mask].reset_index(drop=True)
    valid_product_ratings = product_ratings[mask]

    # Top 10
    top_indices = np.argsort(valid_product_ratings)[-10:][::-1]
    top_products = valid_products.iloc[top_indices]

    ids = []
    titles = []

    for i, (_, product) in enumerate(top_products.iterrows()):
        productDTO = init_productDTO(product)
        ids.append(productDTO.Id)
        titles.append(f'\n{productDTO.Title} (score: {valid_product_ratings[top_indices[i]]})')

    return ids, titles

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
def process_cross_sell_recommendation(raw_interactions: pd.DataFrame, raw_products: pd.DataFrame, enable_outlier_detection: bool = True, outlier_config_name: str = "ecommerce"):
    sparse_product_product_matrix, product_to_recommend_ids, products_for_recommendation = shared.get_cross_sell_interaction_values(raw_interactions, raw_products, enable_outlier_detection, outlier_config_name)

    # Use cosine similarity instead of ALS for cross-sell
    save_cross_sell_recommendations_cosine(sparse_product_product_matrix, product_to_recommend_ids, products_for_recommendation)

def cross_sell_train_model(sparse_user_product):
    now = pd.Timestamp.now()
    sb = StringBuilder()

    model = implicit.als.AlternatingLeastSquares(factors=32, regularization=0.01, alpha=1.0, iterations=25, calculate_training_loss=True) # calculate_training_loss needs to be true if we want to fit_callback work
    model.fit_callback = store_loss(sb)
    model.fit(sparse_user_product, show_progress=False)

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Cross sell model training", sb.__str__())

    return model

def save_cross_sell_recommendations(model: RecommenderBase, sparse_product_product_matrix: csr_matrix, product_to_recommend_ids: pd.Index, products_for_recommendation: pd.DataFrame) -> dict:
    now = pd.Timestamp.now()
    sb = StringBuilder()

    product_to_recommend_ids = product_to_recommend_ids.tolist()
    products_for_recommendation_ids = products_for_recommendation[ID_COL_NAME].tolist()

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
                    product_id = products_for_recommendation_ids[product_index]
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
    
    test_recommendations_for_display = ', '.join([str(product_id) for product_id in recommendations_dict[test_product_for_cross_sell]])

    sb.append(f"Top ten '{test_product_for_cross_sell}' recommendations: {test_recommendations_for_display}\n")

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Storing recommendations for cross sell", sb.__str__())

    return recommendations_dict

def save_cross_sell_recommendations_cosine(sparse_product_product_matrix: csr_matrix, product_to_recommend_ids: pd.Index, products_for_recommendation: pd.DataFrame) -> dict:
    """
    Save cross-sell recommendations using cosine similarity on product-product matrix.
    This is more appropriate than ALS for cross-sell recommendations.
    """
    now = pd.Timestamp.now()
    sb = StringBuilder()

    product_to_recommend_ids = product_to_recommend_ids.tolist()
    products_for_recommendation_ids = products_for_recommendation[ID_COL_NAME].tolist()

    product_indexes_to_filter = get_product_indexes_to_filter(products_for_recommendation)
    sb.append(f'Products to filter count: {len(product_indexes_to_filter)}\n')

    # Calculate cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(sparse_product_product_matrix)
    
    # Set diagonal to 0 to avoid self-recommendations
    np.fill_diagonal(similarity_matrix, 0)
    
    # Set filtered products to 0 similarity
    for idx in product_indexes_to_filter:
        if idx < similarity_matrix.shape[0]:
            similarity_matrix[idx, :] = 0
            similarity_matrix[:, idx] = 0

    recommendations_dict = defaultdict(list)
    
    redis_pipeline = CROSS_SELL_RECOMMENDER_REDIS.pipeline()

    try:
        for i, product_to_recommend_id in enumerate(product_to_recommend_ids):
            if i < similarity_matrix.shape[0]:
                # Get top similar products
                similar_indices = np.argsort(similarity_matrix[i])[::-1]
                products_for_cross_sell = []
                
                for similar_idx in similar_indices:
                    if similar_idx < len(products_for_recommendation_ids):
                        product_id = products_for_recommendation_ids[similar_idx]
                        similarity_score = similarity_matrix[i, similar_idx]
                        
                        # Only include products with meaningful similarity
                        if similarity_score > 0.1 and product_id != product_to_recommend_id:
                            products_for_cross_sell.append(product_id)
                            
                        # Limit to top 20 recommendations
                        if len(products_for_cross_sell) >= 20:
                            break
                
                recommendations_dict[product_to_recommend_id] = products_for_cross_sell
                # Persistent, because it's product to product interactions
                redis_pipeline.set(name=product_to_recommend_id, value=json.dumps(products_for_cross_sell))
        
        redis_pipeline.execute()
    except Exception as ex:
        redis_pipeline.reset()
        raise ex
    
    test_product_for_cross_sell = Settings().TEST_PRODUCT_FOR_CROSS_SELL

    if len(recommendations_dict[test_product_for_cross_sell]) == 0:
        test_product_for_cross_sell = product_to_recommend_ids[0]
    
    test_recommendations_for_display = ', '.join([str(product_id) for product_id in recommendations_dict[test_product_for_cross_sell]])

    sb.append(f"Top ten '{test_product_for_cross_sell}' cross-sell recommendations: {test_recommendations_for_display}\n")

    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("Storing cross-sell recommendations using cosine similarity", sb.__str__())

    return recommendations_dict

#endregion

#region Hyperparameter Optimization

def optimize_als_hyperparameters_test(sparse_user_product_matrix, test_size=0.2, max_combinations=30):
    """
    Test hyperparameter optimization for ALS model.
    This is for testing purposes only - not used in production training.
    
    Args:
        sparse_user_product_matrix: Sparse matrix of user-product interactions
        test_size: Fraction of data to use for testing
        max_combinations: Maximum number of parameter combinations to test
        
    Returns:
        dict: Best parameters and their performance metrics
    """
    from implicit.evaluation import (
        train_test_split, 
        precision_at_k, 
        AUC_at_k,
        mean_average_precision_at_k,
        ndcg_at_k,
        ranking_metrics_at_k
    )
    import random
    from sklearn.model_selection import ParameterGrid
    
    now = pd.Timestamp.now()
    sb = StringBuilder()
    sb.append("=== HYPERPARAMETER OPTIMIZATION TEST ===\n")
    sb.append(f"Matrix shape: {sparse_user_product_matrix.shape}\n")
    sb.append(f"Non-zero interactions: {sparse_user_product_matrix.nnz}\n")
    
    # Split data for evaluation using implicit.evaluation.train_test_split
    # Convert to coo_matrix as required by implicit.evaluation.train_test_split
    from scipy.sparse import coo_matrix
    coo_matrix_data = coo_matrix(sparse_user_product_matrix)
    train_matrix, test_matrix = train_test_split(coo_matrix_data, train_percentage=1-test_size, random_state=123)
    sb.append(f"Train shape: {train_matrix.shape}, Test shape: {test_matrix.shape}\n")
    
    # Define parameter grid
    param_grid = {
        'factors': [550],
        'regularization': [0.01],
        'alpha': [140.0],
        'iterations': [25]
    }
    
    # Generate all parameter combinations using sklearn
    param_combinations = list(ParameterGrid(param_grid))
    
    # Randomly sample combinations to test (to limit computation time)
    test_combinations = random.sample(param_combinations, min(max_combinations, len(param_combinations)))
    sb.append(f"Testing {len(test_combinations)} parameter combinations out of {len(param_combinations)} total\n\n")
    
    best_score = 0
    best_params = None
    results = []
    
    for i, params in enumerate(test_combinations):
        try:
            sb.append(f"Testing combination {i+1}/{len(test_combinations)}: {params}\n")
            
            # Train model with current parameters and early stopping
            model = implicit.als.AlternatingLeastSquares(
                factors=params['factors'],
                regularization=params['regularization'],
                alpha=params['alpha'],
                iterations=params['iterations'],
                calculate_training_loss=True  # Enable for early stopping
            )
            model.fit(train_matrix, show_progress=False)
            
            # Evaluate model using implicit.evaluation functions
            # All functions now require explicit parameter names as per documentation
            precision_10 = precision_at_k(model, train_matrix, test_matrix, K=10, show_progress=False, num_threads=1)
            precision_5 = precision_at_k(model, train_matrix, test_matrix, K=5, show_progress=False, num_threads=1)
            
            # Add additional evaluation metrics
            auc_10 = AUC_at_k(model, train_matrix, test_matrix, K=10, show_progress=False, num_threads=1)
            map_10 = mean_average_precision_at_k(model, train_matrix, test_matrix, K=10, show_progress=False, num_threads=1)
            ndcg_10 = ndcg_at_k(model, train_matrix, test_matrix, K=10, show_progress=False, num_threads=1)
            
            # Calculate composite score using only official implicit.evaluation metrics
            # Ben Fred's metrics are well-chosen for recommendation systems
            composite_score = 0.3 * precision_10 + 0.25 * precision_5 + 0.2 * auc_10 + 0.15 * map_10 + 0.1 * ndcg_10
            
            result = {
                'params': params,
                'precision@5': precision_5,
                'precision@10': precision_10,
                'auc@10': auc_10,
                'map@10': map_10,
                'ndcg@10': ndcg_10,
                'composite_score': composite_score
            }
            results.append(result)
            
            sb.append(f"  Precision@5: {precision_5:.4f}, Precision@10: {precision_10:.4f}\n")
            sb.append(f"  AUC@10: {auc_10:.4f}, MAP@10: {map_10:.4f}, NDCG@10: {ndcg_10:.4f}\n")
            sb.append(f"  Composite Score: {composite_score:.4f}\n")
            
            if composite_score > best_score:
                best_score = composite_score
                best_params = params
                sb.append(f"  *** BEST SCORE! ***\n")
            
        except Exception as e:
            sb.append(f"  Error with params {params}: {str(e)}\n")
            continue
    
    # Sort results by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    sb.append(f"\n=== OPTIMIZATION RESULTS ===\n")
    sb.append(f"Best parameters: {best_params}\n")
    sb.append(f"Best composite score: {best_score:.4f}\n")
    sb.append(f"Total combinations tested: {len(results)}\n")
    
    # Show top 5 results
    sb.append(f"\nTop 5 parameter combinations:\n")
    for i, result in enumerate(results[:5]):
        sb.append(f"{i+1}. Score: {result['composite_score']:.4f} - {result['params']}\n")
    
    sb.append(shared.get_duration_message(now))
    Emailing().send_email_and_log_info("ALS Hyperparameter Optimization Test", sb.__str__())
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results,
        'test_info': {
            'total_combinations': len(results),
            'test_size': test_size,
            'matrix_shape': sparse_user_product_matrix.shape
        }
    }

#endregion

#region Shared

def get_products_for_display(product_ids: list[int]) -> str:
    product_ids = [str(product_id) for product_id in product_ids]
    return ', '.join(product_ids)

def init_productDTO(row: pd.Series):
    return ProductDTO(
        Id=row[ID_COL_NAME], 
        Stock=row[STOCK_COL_NAME], 
        Status=row[STATUS_COL_NAME], 
        Title=row[TITLE_COL_NAME]
    )

# https://github.com/benfred/implicit/issues/281
def store_loss(output: StringBuilder): 
    def inner(iteration, elapsed, loss): 
        output.append(f'Loss {loss:.6f}\n') 

    return inner

def get_product_indexes_to_filter(products: pd.DataFrame) -> pd.Index:
    return products.loc[
        (products[STOCK_COL_NAME] == 0) |
        (products[STATUS_COL_NAME] != 'Published')
    ].index

#endregion