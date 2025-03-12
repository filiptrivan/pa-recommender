from collections import defaultdict
import pandas as df
from pandas import DataFrame, Series
import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.recommender_base import RecommenderBase
import numpy as np
import implicit
from scipy.sparse import csr_matrix

from DTO.ProductDTO import ProductDTO
from utils.shared import *

SKU = 0
STOCK = 1
STATUS = 2
VISIBILITY = 3
ACTIVE = 4

def get_recommendation_result():
    all_products = load_excel_list('../pa-data/AllProducts.xlsx')
    save_interaction_values("../pa-data/Interactions.xlsx", all_products)

    Y = get_data()
    sparse_user_product_matrix = get_sparse_user_product_matrix(Y)

    products = load_csv_df('Products.csv')
    product_indexes_to_filter = get_product_indexes_to_filter(products)
    user_ids = load_csv_df('Users.csv')

    model = train_the_model(sparse_user_product_matrix)

    return get_recommendation_result_dict(model, sparse_user_product_matrix, user_ids, products, filter_items=product_indexes_to_filter)

def get_sparse_user_product_matrix(Y):
    sparse_product_user = csr_matrix(Y)
    # sparse_product_user = bm25_weight(sparse_product_user, K1=100, B=0.8)
    # get the transpose since the most of the functions in implicit expect (user, product) sparse matrices instead of (product, user)
    sparse_user_product = sparse_product_user.T.tocsr()
    return sparse_user_product

def get_product_indexes_to_filter(products: DataFrame):
    result = [
        idx for idx, x in products.iterrows()
        if x[STOCK] == 0 or x[STATUS] != 'Published' or x[VISIBILITY] != 'Public' or x[ACTIVE] != True
    ]
    return result

def train_the_model(sparse_user_product):
    # https://github.com/benfred/implicit/issues/281
    model = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, alpha=1.0, iterations=128, calculate_training_loss=True)
    model.fit(sparse_user_product, show_progress=True)
    return model

def get_recommendation_result_dict(model: RecommenderBase, sparse_user_product_matrix: csr_matrix, user_ids: DataFrame, products: DataFrame, filter_items) -> dict:
    recommendations_dict = {}
    
    recommendations_dict['top_ten_overall_recommendations'] = get_top_overall_recommendations(sparse_user_product_matrix, products)
    batch_size = 1000
    to_generate = np.arange(len(user_ids))
    
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx : startidx + batch_size]
        product_indexes, scores = model.recommend(batch, sparse_user_product_matrix[batch], filter_already_liked_items=False, filter_items=filter_items)
        for i, user_index in enumerate(batch):
            user_id = user_ids.iloc[user_index].values[0]
            products_for_recommendation = []
            for product_index, score in zip(product_indexes[i], scores[i]):
                product: Series = products.iloc[product_index]
                dto = init_product(product)
                products_for_recommendation.append(dto.__dict__)
            recommendations_dict[user_id] = products_for_recommendation
    
    return recommendations_dict

def get_top_overall_recommendations(sparse_user_product_matrix: csr_matrix, products: DataFrame) -> list[dict]:
    result: list[dict] = []

    product_ratings = sparse_user_product_matrix.sum(axis=0)
    product_ratings = np.asarray(product_ratings).flatten()

    product_counts = sparse_user_product_matrix.getnnz(axis=0)

    avg_interactions = np.divide(
        product_ratings, 
        product_counts, 
        out=np.zeros_like(product_ratings, dtype=float), 
        where=product_counts != 0
    )

    invalid_indices = get_product_indexes_to_filter(products)
    valid_products = products.drop(index=invalid_indices)
    valid_avg_interactions = np.delete(avg_interactions, invalid_indices)

    top_10_indices = np.argsort(valid_avg_interactions)[-10:][::-1]

    top_10_ratings = valid_products.iloc[top_10_indices]

    for idx, row in top_10_ratings.iterrows():
        dto = init_product(row)
        result.append(dto.__dict__)

    return result

def init_product(row: Series):
    return ProductDTO(Id=row[SKU], Stock=int(row[STOCK]), Status=row[STATUS], Visibility=row[VISIBILITY], Active=bool(row[ACTIVE]))

def benchmark_accuracy(sparse_user_product): 
    output = defaultdict(list) 

    def store_loss(name): 
        def inner(iteration, elapsed, loss): 
            print(f"model {name} iteration {iteration} loss {loss:.5f}") 
            output[name].append(loss) 

        return inner 

    for steps in [2, 3, 4]: 
        model = implicit.als.AlternatingLeastSquares( 
            factors=100, 
            use_gpu=False, 
            regularization=0.1, 
            iterations=25, 
            calculate_training_loss=True, 
        ) 
        model.cg_steps = steps 
        model.fit_callback = store_loss(f"cg{steps}") 
        model.fit(sparse_user_product) 