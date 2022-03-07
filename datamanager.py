import pandas as pd
import os
from termcolor import colored
from copy import deepcopy
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import functools

CAN_BE_DROPPED = {
    'item_asset': ['asset_upload_number'],
    'item_price': ['price_upload_number'],
    'user_age': ['age_upload_number'],
    # 'interactions': ['has_interaction'],
    'item_subclass': ['item_category_belonging'],
    'user_region': ['user_region_belonging']
}

PRUNE_CLAUSE = {
    'interactions': {
        'col_name': 'has_interaction',
        'condition': 'val equals zero'
    },
    'item_subclass': {
        'col_name': 'item_category_belonging',
        'condition': 'val equals zero'
    },
    'user_region': {
        'col_name': 'user_region_belonging',
        'condition': 'val equals zero'
    }
}

COLUMNS_PERMUTATION = {
    'interactions': {
        'row': 'user_id',
        'col': 'item_id',
        'data': 'has_interaction'
    },
    'item_asset': {
        'row': 'item_id',
        'col': 'asset_upload_number',
        'data': 'item_asset'
    },
    'item_price': {
        'row': 'item_id',
        'col': 'price_upload_number',
        'data': 'item_price'
    },
    'item_subclass': {
        'row': 'item_id',
        'col': 'item_category_id',
        'data': 'item_category_belonging'
    },
    'user_age': {
        'row': 'user_id',
        'col': 'age_upload_number',
        'data': 'user_age'
    },
    'user_region': {
        'row': 'user_id',
        'col': 'region_id',
        'data': 'user_region_belonging'
    }
}

DROP_TABLES_AFTER_MERGE = [
    'item_asset',
    'item_price',
    'item_subclass',
    'user_age',
    'user_region'
]


class Checkers:
    @staticmethod
    def check_validity(func):
        @functools.wraps(func)
        def wrapper_check_validity(*args, **kwargs):
            if not isinstance(args[0], DataManager):
                raise RuntimeError('Incorrect usage of this wrapper')
            if args[0].data is None:
                raise RuntimeError('Call setup() first..')
            return func(*args, **kwargs)
        return wrapper_check_validity


class DataManager:
    def __init__(self, data_dir_path: str, train_split: float = 0.8):
        self.data_dir_path = data_dir_path
        self.train_split = train_split

        self.data = None
        self.train_interactions = None
        self.test_interactions = None

    def setup(self, verbose: bool = True):
        self.data = self._load()
        self.data, num_prunes = self._prune()
        self.data = self._drop_unnecessary_cols()
        self.data, num_nan = self._dropna()

        self.data['users'] = deepcopy(
            self.data['user_age']
        ).merge(self.data['user_region'], on='user_id', how='inner')

        self.data['items'] = deepcopy(
            self.data['item_subclass']
        ).merge(
            self.data['item_price'], on='item_id', how='inner'
        ).merge(self.data['item_asset'], on='item_id', how='inner')

        if verbose:
            print(colored('Number of prunes:\n<table name>.<col name>: <num prunes>', 'red'))
            for key in num_prunes.keys():
                print('{}:\t{}'.format(key, num_prunes[key]))

            print(colored('Number of NAN rows:\n<tab name>\n<col name>\t<num NAN>\ndtype:  <dtype>', 'red'))
            for key in num_nan.keys():
                print(colored(key, 'red') + ':')
                print('{}'.format(num_nan[key]))

            print('\nDropping unnecessary tables after merge...\n')

        for key in DROP_TABLES_AFTER_MERGE:
            self.data.pop(key)

        self.train_interactions, self.test_interactions = train_test_split(
            self.data['interactions'],
            # stratify=self.data['interactions']['user_id'],
            train_size=self.train_split,
            random_state=42
        )

    def describe(self, describe: bool = True, info: bool = False, head: bool = False):
        for key in self.data.keys():
            print(colored(key, 'red'))
            if describe:
                print('{}\n\n'.format(self.data[key].describe()))
            if info:
                print('{}\n\n'.format(self.data[key].info()))
            if head:
                print('{}\n\n'.format(self.data[key].head()))

    @Checkers.check_validity
    def get_train_data(self):
        train_data = {
            'interactions': self.train_interactions,
            'users': deepcopy(self.data['users']),
            'items': deepcopy(self.data['items'])
        }
        return train_data

    @Checkers.check_validity
    def get_test_data(self):
        test_data = {
            'interactions': self.test_interactions
        }
        return test_data

    def _load(self) -> Dict[str, pd.DataFrame]:
        self.data = dict.fromkeys([el.split('.')[0] for el in os.listdir(self.data_dir_path)])
        for key in self.data.keys():
            self.data[key] = pd.read_csv(os.path.join(self.data_dir_path, '{}.csv'.format(key)))
            self.data[key] = self.data[key].rename(columns=COLUMNS_PERMUTATION[key])

        return self.data

    def _drop_unnecessary_cols(self) -> Dict[str, pd.DataFrame]:
        for key in CAN_BE_DROPPED.keys():
            self.data[key] = self.data[key].drop(columns=CAN_BE_DROPPED[key], axis=1)
        return self.data

    def _prune(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
        num_prunes = dict()
        for key in PRUNE_CLAUSE:
            col_name = PRUNE_CLAUSE[key]['col_name']
            condition = PRUNE_CLAUSE[key]['condition']
            if condition == 'val equals zero':
                num_prunes['{}.{}'.format(key, col_name)] = len(self.data[key][self.data[key][col_name] == 0])
                self.data[key] = self.data[key][self.data[key][col_name] != 0]

        return self.data, num_prunes

    def _dropna(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
        num_nan = dict()
        for key in self.data.keys():
            num_nan[key] = self.data[key].isna().sum()
            self.data[key] = self.data[key].dropna()

        return self.data, num_nan



