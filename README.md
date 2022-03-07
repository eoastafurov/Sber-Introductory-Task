# Sber Introductory Task

### Aim
The online store has accumulated data on the interaction of buyers with goods for several months. 
The goal is to recommend a product that will cause the buyer to interact with it.

1. The goal is to recommend each customer a list of 10 potentially relevant products.
2. To assess the quality of recommendations, the MAP@10 metric is used
3. You can use any recommender system algorithm (collaborative filtering, content-based, hybrid, etc.)
4. To split the dataset into train and test, use a random split 80/20


### Data


* `interactions.csv` — the file stores data on the interaction of goods and customers. Among the data there are "cold" goods and buyers. The `row` column stores the customer identifiers. In the `col` column are product identifiers. In the `data` column - the value of the interaction.

* `item_asset.csv` - the file stores the qualitative characteristics of the product. `row` - item identifier, `data` - characteristic value. `col` - serial number of the feature when uploading data (does not make sense, you can get rid of this column)


* `item_price.csv` - the file stores the price of the item (already normalized). `row` - product identifier, `data` - normalized price value. `col` - serial number of the feature when uploading data (does not make sense, you can get rid of this column)


* `item_subclass.csv` - the file stores the values of the categories to which the product belongs. `row` - product identifier, `col` - category number, `data` - attribute of relation to the category


* `user_age.csv` - the file stores data on the age of users. `row` - user identifier, `data` - age value (already normalized), `col` - feature serial number when uploading data (does not make sense, you can get rid of this column)


* `user_region.csv` - the file stores the one-hot encoded values of the user's region. `row` - user ID, `col` - number of one-hot feature of the region, `data` - feature of the region.


----

## Usage

You can see usage examples in `examples.ipynb`.

##### TL;DR
* Models API:
```python
model = ModelName(...)
model.fit(train_dataframe)
model.predict(user_id, items_to_ignore, topn)
```

* Evaluating
```python
ev = ModelEvaluator(model, test_data, full_data, train_data)
global_metrics, detailed_results = ev.evaluate_model(num_values=N)
```