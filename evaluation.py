import pandas as pd
import random
from typing import Dict, Set, Optional, List, Tuple
from tqdm import tqdm

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 10
"""
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS
is the const param that defines number of
non interacted items in sample.
"""


class DataUtils:
    @staticmethod
    def get_items_interacted(
            interactions_df: pd.DataFrame,
            person_id: int
    ) -> Set[int]:
        try:
            interacted_items = interactions_df.loc[person_id]['item_id']
        except KeyError:
            interacted_items = ()

        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    @staticmethod
    def get_non_interacted_items(
            full_data: Dict[str, pd.DataFrame],
            person_id: int
    ) -> Set[int]:
        interacted_items = DataUtils.get_items_interacted(full_data['interactions'], person_id)
        all_items = set(full_data['items']['item_id'])
        non_interacted_items = all_items - interacted_items
        return non_interacted_items

    @staticmethod
    def sample_not_interacted_items(
            full_data: Dict[str, pd.DataFrame],
            person_id: int,
            sample_size: int,
            seed: int = 42
    ) -> Set[int]:
        non_interacted_items = DataUtils.get_non_interacted_items(full_data, person_id)

        random.seed(seed)
        non_interacted_items_sample = random.sample(list(non_interacted_items), sample_size)
        return set(non_interacted_items_sample)

    @staticmethod
    def verify_hit_top_n(
            item_id: int,
            recommended_items: List[int],
            topn: int
    ) -> Tuple[int, int]:
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except StopIteration:
            index = -1
        hit = int(topn > index >= 0)
        return hit, index

    @staticmethod
    def apk(
            actual: List[int],
            predicted: List[int],
            topn: int
    ) -> float:
        if not actual:
            return 0.0

        if len(predicted) > topn:
            predicted = predicted[:topn]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(actual), topn)


class ModelEvaluator:
    def __init__(self, model, test_data, full_data, train_data):
        self.model = model
        self.test_data = test_data
        self.train_data = train_data
        self.full_data = full_data

    def evaluate_model_for_user(
            self,
            person_id: int
    ) -> Dict[str, float]:
        """
        Get metrics for explicit user,
        provided with id
        """
        interacted_values_testset = self.test_data['interactions'].loc[person_id]
        if type(interacted_values_testset['item_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['item_id'])
        else:
            person_interacted_items_testset = {int(interacted_values_testset['item_id'])}

        interacted_items_count_testset = len(person_interacted_items_testset)

        person_recs_df = self.model.predict(
            person_id,
            items_to_ignore=DataUtils.get_items_interacted(
                self.train_data['interactions'],
                person_id
            ),
            topn=100
        )

        hits_at_10_count = 0
        ap10 = DataUtils.apk(
            actual=list(interacted_values_testset),
            predicted=list(person_recs_df['item_id']),
            topn=10
        )

        # For each item the user has interacted in test set
        for i, item_id in enumerate(person_interacted_items_testset):
            non_interacted_items_sample = DataUtils.sample_not_interacted_items(
                full_data=self.full_data,
                person_id=person_id,
                sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                seed=42
            )

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union({item_id})
            valid_recs_df = person_recs_df[person_recs_df['item_id'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['item_id'].values
            hit_at_10, index_at_10 = DataUtils.verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
        person_metrics = {
            'hits@10_count': hits_at_10_count,
            'interacted_testset_count': interacted_items_count_testset,
            'recall@10': recall_at_10,
            'ap@10': ap10
        }

        return person_metrics

    def evaluate_model(
            self,
            num_values: Optional[int] = None
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Get global metrics for model along testset
        :param num_values: number of test data rows.
            P.S. Fill with None if you want to
            test on whole testset
        :return: Global metrics dictionary and dataframe with details
        """
        people_metrics = []
        tests = list(self.test_data['interactions'].index.unique().values)
        if num_values is not None:
            tests = tests[:num_values]

        for idx, person_id in tqdm(enumerate(tests)):
            person_metrics = self.evaluate_model_for_user(person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)

        detailed_results_df = pd.DataFrame(
            people_metrics
        ).sort_values('interacted_testset_count', ascending=False)

        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_testset_count'].sum())

        mean_apk10 = detailed_results_df['ap@10'].sum() / float(
            detailed_results_df['interacted_testset_count'].sum()
        )

        global_metrics = {
            'recall@10': global_recall_at_10,
            'map@10': mean_apk10
        }

        return global_metrics, detailed_results_df
