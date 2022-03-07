import functools
import os
import pickle
from typing import List
import pandas as pd


class Checkers:
    @staticmethod
    def check_fitted(func):
        """
        Checks if method `fit(...)` was called and
        necessary tables if self scope was generated
        """
        @functools.wraps(func)
        def wrapper_check_fitted(*args, **kwargs):
            if not isinstance(args[0], PopularityModel):
                raise RuntimeError('Incorrect usage of this wrapper!')
            if args[0].popularity_df is None:
                raise RuntimeError('Call fit(...) first!')
            return func(*args, **kwargs)
        return wrapper_check_fitted


class PopularityModel:
    """
    Common (and usually hard-to-beat) baseline approach.
    This model is not actually personalized - it simply
    recommends to a user the most popular items that
    the user has not previously consumed
    """
    def __init__(self, items_df: pd.DataFrame):
        self.popularity_df = None
        self.items_df = items_df

    @Checkers.check_fitted
    def save(self, path: str):
        """
        Save model fields to pickle format
        :param path: full path, ends with .pkl
        :return: dict with params
        """
        model_dict = {
            'items_df': self.items_df,
            'popularity_df': self.popularity_df
        }
        with open(path, 'w+b') as f:
            pickle.dump(model_dict, f)
        return model_dict

    @staticmethod
    def load_from_file(path: str):
        """
        Load model configuration and weights
        from file.
        Also perform one test
        """
        if not os.path.exists(path):
            raise FileNotFoundError

        with open(path, 'rb') as f:
            model_dict = pickle.load(f)

        model = PopularityModel(items_df=model_dict['items_df'])
        model.popularity_df = model_dict['popularity_df']

        model.predict(
            user_id=None,
            items_to_ignore=[],
            topn=1
        )

        return model

    def fit(self, train_interactions_df: pd.DataFrame):
        self.popularity_df = train_interactions_df.groupby(
            'item_id'
        ).sum().sort_values(
            by='has_interaction',
            ascending=False
        ).reset_index()

        return self

    @Checkers.check_fitted
    def predict(self, user_id, items_to_ignore: List[int], topn: int):
        recommendations_df = self.popularity_df[
            ~self.popularity_df['item_id'].isin(items_to_ignore)
        ].sort_values(
            'has_interaction',
            ascending=False
        ).head(topn)

        return recommendations_df
