import functools
import os.path
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pickle
import random


class Checkers:
    @staticmethod
    def check_fitted(func):
        """
        Checks if method `fit(...)` was called and
        necessary tables if self scope was generated
        """
        @functools.wraps(func)
        def wrapper_check_fitted(*args, **kwargs):
            if not isinstance(args[0], CollaborativeLatentFactorSVDModel):
                raise RuntimeError('Incorrect usage of this wrapper!')
            if args[0].cf_preds_df is None or args[0].users_ids is None:
                raise RuntimeError('Call fit(...) first!')
            return func(*args, **kwargs)
        return wrapper_check_fitted


class CollaborativeLatentFactorSVDModel:
    """
    This method is model-based collaborative filtering.
    Latent factor model compresses user-item matrix into a
    low-dimensional representation in terms of latent factors.
    """
    def __init__(self, number_of_factors: int = 15):
        """
        :param number_of_factors: Number of singular
            values and singular vectors to compute
        """
        self.number_of_factors = number_of_factors
        self.cf_preds_df = None
        self.users_ids = None

    def _hyperparams(self) -> Dict:
        return {'number_of_factors': self.number_of_factors}

    @Checkers.check_fitted
    def save(self, path: str):
        """
        Save model fields to file in
        pickle format
        :param path: path to file,
            ends with .pkl
        """
        model_dict = {
            'cf_preds_df': self.cf_preds_df,
            'users_ids': self.users_ids,
            'hyperparams': self._hyperparams()
        }
        with open(path, 'w+b') as f:
            pickle.dump(model_dict, f)
        return model_dict

    @staticmethod
    def load_from_file(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError

        with open(path, 'rb') as f:
            model_dict = pickle.load(f)

        model = CollaborativeLatentFactorSVDModel()
        model.users_ids = model_dict['users_ids']
        model.cf_preds_df = model_dict['cf_preds_df']
        model.number_of_factors = model_dict['hyperparams']['number_of_factors']

        random.seed(42)
        model.predict(
            user_id=random.choice(model.users_ids),
            items_to_ignore=[],
            topn=1
        )

        return model

    def fit(self, train_interactions_df: pd.DataFrame):
        users_items_pivot_matrix_df = train_interactions_df.pivot(
            index='user_id',
            columns='item_id',
            values='has_interaction'
        ).fillna(0)
        users_items_pivot_matrix = users_items_pivot_matrix_df.values
        self.users_ids = list(users_items_pivot_matrix_df.index)

        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=self.number_of_factors)
        sigma = np.diag(sigma)

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        all_user_predicted_ratings_norm = (
            all_user_predicted_ratings -
            all_user_predicted_ratings.min(initial=np.inf)
        ) / (
            all_user_predicted_ratings.max(initial=-np.inf) -
            all_user_predicted_ratings.min(initial=np.inf)
        )

        self.cf_preds_df = pd.DataFrame(
            all_user_predicted_ratings_norm,
            columns=users_items_pivot_matrix_df.columns,
            index=self.users_ids
        ).transpose()

        return self

    @Checkers.check_fitted
    def predict(self, user_id, items_to_ignore: List[int], topn: int) -> pd.DataFrame:
        if user_id not in self.users_ids:
            random.seed(42)
            user_id = random.choice(self.users_ids)

        sorted_user_predictions = self.cf_preds_df[user_id].sort_values(
            ascending=False
        ).reset_index().rename(columns={user_id: 'recommend_strength'})

        recommendations_df = sorted_user_predictions[
            ~sorted_user_predictions['item_id'].isin(items_to_ignore)
        ].sort_values('recommend_strength', ascending=False).head(topn)

        return recommendations_df
