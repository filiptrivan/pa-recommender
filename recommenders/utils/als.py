from collections import defaultdict
import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.recommender_base import RecommenderBase
import numpy as np
import implicit
from scipy.sparse import csr_matrix

from recommenders.utils.shared import *

SKU = 0
STOCK = 1
STATUS = 2
VISIBILITY = 3
ACTIVE = 4

def get_recommendation_result():
    all_products = load_excel_list('../../pa-data/AllProducts.xlsx')
    save_interaction_values("../../pa-data/Interactions.xlsx", all_products)

    Y = get_data()
    sparse_user_product_matrix = get_sparse_user_product_matrix(Y)

    products = load_csv_list('Products.csv')
    product_names = np.array([row[0] for row in products])
    product_indexes_to_filter = get_product_indexes_to_filter(products)
    user_ids = load_csv_list('Users.csv')[0]

    model = train_the_model(sparse_user_product_matrix)

    return get_recommendation_result_dict(model, sparse_user_product_matrix, user_ids, product_names, filter_items=product_indexes_to_filter)

def get_sparse_user_product_matrix(Y):
    sparse_product_user = csr_matrix(Y)
    # sparse_product_user = bm25_weight(sparse_product_user, K1=100, B=0.8)
    # get the transpose since the most of the functions in implicit expect (user, product) sparse matrices instead of (product, user)
    sparse_user_product = sparse_product_user.T.tocsr()
    return sparse_user_product

def get_product_indexes_to_filter(products):
    return [
        idx for idx, x in enumerate(products)
        if x[STOCK] == '0' or x[STATUS] != 'Published' or x[VISIBILITY] != 'Public' or x[ACTIVE] != '1'
    ]

def train_the_model(sparse_user_product):
    # https://github.com/benfred/implicit/issues/281
    model = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, alpha=1.0, iterations=128, calculate_training_loss=True)
    model.fit(sparse_user_product, show_progress=True)
    return model

def get_recommendation_result_dict(model: RecommenderBase, sparse_user_product_matrix: csr_matrix, user_ids, product_names, filter_items):
    recommendations_dict = {}
    
    recommendations_dict['top_ten_overall_recommendations'] = get_top_overall_recommendations(sparse_user_product_matrix, product_names)

    batch_size = 1000
    to_generate = np.arange(len(user_ids))

    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx : startidx + batch_size]
        product_indexes, scores = model.recommend(batch, sparse_user_product_matrix[batch], filter_already_liked_items=False, filter_items=filter_items)
        for i, userid in enumerate(batch):
            user_id = user_ids[userid]
            product_ids_for_recommendation = []
            for product_index, score in zip(product_indexes[i], scores[i]):
                product_name = product_names[product_index]
                product_ids_for_recommendation.append(product_name)
            recommendations_dict[user_id] = product_ids_for_recommendation
    
    return recommendations_dict

def get_top_overall_recommendations(sparse_user_product_matrix: csr_matrix, product_names):
    product_ratings = sparse_user_product_matrix.sum(axis=0)
    product_ratings = np.asarray(product_ratings).flatten()

    product_counts = sparse_user_product_matrix.getnnz(axis=0)

    avg_interactions = np.divide(
        product_ratings, 
        product_counts, 
        out=np.zeros_like(product_ratings, dtype=float), 
        where=product_counts != 0
    )

    top_10_indices = np.argsort(avg_interactions)[-10:][::-1]
    top_10_ratings = product_names[top_10_indices]

    return top_10_ratings

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