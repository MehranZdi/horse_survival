# horse_survival
Horse Survival Using SVM on [Horse Colic](https://archive.ics.uci.edu/dataset/47/horse+colic) Dataset.

## Explanation of Crucial Parts:

### Dealing with Missing Values:

All missed values should be taken care of. The method which has been used to deal with missed values is to fill them by the most frequent value in each feature.
The following code will fill the missing values in both train and test set and after running the code the datasets will have no longer any missed values.

``` python
def miss_handler(data):
    imputer = SimpleImputer(strategy='most_frequent')
    data = pd.DataFrame(
        imputer.fit_transform(data), columns=data.columns).astype(data.dtypes.to_dict())
    return data

x_train = miss_handler(x_train)
x_test = miss_handler(x_test)
```
### Encoding and Scaling:

Two vital things should be done, the first one is that the numerical data have to be scaled to have mean 0 and variance 1.
The second one is to encode the categorical features since our classifier cannot process data types except ‘integer’ or ‘float’. To do so, we have used ‘OrdinalEncoder’ for our ordinal-categorical features and have used ‘OneHotEncoder’ for nominal-categorical features.
All the processes can be done in a single shot with the help of Sci-kit Learn pipelines and transformers.
Different classes have been implemented to take care of each feature type as follows:
```python
class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_categories):
        self.ordinal_categories = ordinal_categories
        self.ordinal_encoder = OrdinalEncoder(categories=[self.ordinal_categories[f] for f in self.ordinal_categories])

    def fit(self, X, y=None):
        return self.ordinal_encoder.fit(X)

    def transform(self, X):
        return self.ordinal_encoder.transform(X)


class NominalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.onehot_encoder = OneHotEncoder(drop='first')

    def fit(self, X, y=None):
        return self.onehot_encoder.fit(X)

    def transform(self, X):
        return self.onehot_encoder.transform(X)


train_preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', NumericalTransformer(), num_feat.columns),  # StandardScaler for numerical features
        ('ordinal', OrdinalTransformer(ordinal_categories), list(ordinal_categories.keys())),  # ordinal transformer
        ('nominal', NominalTransformer(), nominal_cats)  # nominal transformer
    ]
)

test_preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', NumericalTransformer(), num_feat.columns),
        ('ordinal', OrdinalTransformer(ordinal_categories), list(ordinal_categories.keys())),
        ('nominal', NominalTransformer(), nominal_cats)
    ]
)


train_pipeline = Pipeline(steps=[('preprocessor', train_preprocessor)])
test_pipeline = Pipeline(steps=[('preprocessor', test_preprocessor)])
```
Consider that we first should specify our features:
```python
num_feat = x_train.select_dtypes(include=['float64', 'int64'])


ordinal_categories = {
    'peripheral_pulse': ['normal', 'increased', 'reduced', 'absent'],
    'capillary_refill_time': ['more_3_sec', '3', 'less_3_sec'],
    'peristalsis': ['hypomotile', 'normal', 'hypermotile', 'absent'],
    'abdominal_distention': ['none', 'slight', 'moderate', 'severe'],
    'nasogastric_tube': ['none', 'slight', 'significant'],
    'nasogastric_reflux': ['none', 'less_1_liter', 'more_1_liter'],
    'rectal_exam_feces': ['normal', 'increased', 'decreased', 'absent'],
    'abdomen': ['normal', 'other', 'firm', 'distend_small', 'distend_large'],
    'abdomo_appearance': ['clear', 'cloudy', 'serosanguious']
}

nominal_cats = ['temp_of_extremities', 'mucous_membrane', 'pain']
```
To put all transformers together and apply them on both train and test set:
```python
x_train_transformed = train_pipeline.fit_transform(x_train)
x_test_transformed = test_pipeline.fit_transform(x_test)
num_ord_feature_names = (list(make_column_selector(dtype_include=['float64', 'int64'])(x_train)) +
                         list(ordinal_categories.keys()))


nom_feature_names = []
nominal_encoder = train_pipeline.named_steps['preprocessor'].transformers_[2][1].onehot_encoder
for i, col in enumerate(nominal_cats):
    categories = nominal_encoder.categories_[i][1:]
    nom_feature_names.extend([f'{col}_{cat}' for cat in categories])

feature_names = num_ord_feature_names + nom_feature_names

x_train_transformed = pd.DataFrame(x_train_transformed, columns=feature_names)
x_test_transformed = pd.DataFrame(x_test_transformed, columns=feature_names)
```
### Defining SVM classifiers
#### Classifier Comparator Funciton with Different Kernels
At first, we're going to use SVM classifiers with different kernel, C values, Gamma and Degree(for 'rbf' and 'poly' kernels). The function below make classifires and store the result in a dictionary.
```python
def kernel_comparator(clf_num, c, kernel, x_train, y_train, x_test, y_test, degree=-1, gamma=-1):
    history = dict()
    if gamma > 0 and degree < 0 :
        clf = SVC(C = c, kernel=kernel, gamma=gamma)
        degree='None'

    elif degree > 0 and gamma >0: 
        clf = SVC(C = c, kernel=kernel, degree=degree, gamma=gamma)
    
    else: 
        clf = SVC(C = c, kernel=kernel)
        degree = 'None'
        gamma = 'None'
    
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    history[clf_num] = [kernel, train_acc, test_acc, precision, recall, c, degree, gamma]
    return history
```
Turn to call the function:
```python
clf0 = kernel_comparator(0, 25, 'linear', x_train_transformed, y_train, x_test_transformed, y_test)
clf1 = kernel_comparator(1, 1, 'poly', x_train_transformed, y_train, x_test_transformed, y_test, degree=3, gamma=30)
clf2 = kernel_comparator(2, 35, 'poly', x_train_transformed, y_train, x_test_transformed, y_test, degree=4, gamma=5)
clf3 = kernel_comparator(3, 15, 'rbf', x_train_transformed, y_train, x_test_transformed, y_test, gamma=20)
clf4 = kernel_comparator(4, 2, 'rbf', x_train_transformed, y_train, x_test_transformed, y_test, gamma=0.5)
```
And the result, after running the following code will be:
```python
evals_list = [clf0, clf1, clf2, clf3, clf4]
histories = {}
counter = 0
for dict in evals_list:
    for key in dict.keys():
        clf_num = 'clf' + ' ' + str(key)
        histories[clf_num] = dict[key]
        counter += 1

model_result = pd.DataFrame(histories.values(), index=histories.keys(), columns=['kernel', 'train_acc', 'test_acc', 'precision', 'recall', 'c', 'degree', 'gamma'])
model_result
```
![Evaluation result for different kernels](https://github.com/MehranZdi/horse_survival/blob/main/evaluation_result_dff_kernels.png)

#### Classifier Comparator Function with Linear Kernel and Different C Values
We want to make five objects of SVC() with linear kernel and make a comparison between them. For each classifier, we have used different parameter C. The following function tries to generate random values for C. Note that it has been considered that the generated values have an increasing trend.
```python
def make_random(prev_num=0):
    flag = True
    while flag:
        num = random.random()
        if num > prev_num:
            flag = False
        else:
            num = prev_num
    return num
```
The following function, ‘comparator’, takes datasets and applies SVC() on them and saves the results in a python dictionary called ‘histories’ which we need further.

```python
def linear_comparator(x_train, y_train, x_test, y_test):
    histories = {}
    c = 0
    for i in range(5):
        c = make_random(c)
        clf = SVC(C = c, kernel='linear')
        clf.fit(x_train, y_train)
        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        y_pred = clf.predict(x_test_transformed)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        histories[f'clf{i}'] = [train_acc, test_acc, precision, recall, c, clf]
    return histories
```
### Show the evaluation results
```python
def show_eval_metrics():
    counter = 0
    metrics = {}
    for item in score_history:
        clf_num = 'clf' + ' ' + str(counter)
        metrics[clf_num] = [score_history[item][0], score_history[item][1], score_history[item][2], score_history[item][3]]
        print(f'{clf_num}:\nTrain accuracy: {score_history[item][0]},\nTest accuracy: {score_history[item][1]}\nPrecision: {score_history[item][2]}\nRecall: {score_history[item][3]} \nc: {score_history[item][4]}')
        print('-----------------------------------------------------')
        counter += 1

    return metrics
```
The following code makes a dataframe of the evaluation metrics, accuracy, precision and recall for both train and test set:
```python
model_result = pd.DataFrame(eval_metrics.values(), index=eval_metrics.keys(), columns=['train_acc', 'test_acc', 'precision', 'recall'])
```
The result is:

![Linear Evaluation result](https://github.com/MehranZdi/horse_survival/blob/main/linear_evaluation_result.png)

### Confusion Matrix
And last but not least, the confusion matrix is a crucial part of each machine learning model to get information about the evaluation metrics. First, we have implemented a function to extract the best classifier out of all classifiers we have, based on all evaluation metrics. And then a confusion matrix has been plotted.
```python
def choose_clf(choices=eval_metrics):
    for i in range(len(choices) - 1):
        first_clf = list(choices.keys())[i]
        second_clf = list(choices.keys())[i+1]

        if (choices[first_clf][0] > choices[second_clf][0]) and (choices[first_clf][1] > choices[second_clf][1]):
            i += 1
        else:
            first_clf = second_clf
            i += 1
    
    return first_clf
```
Confusion matrix plot:
```python
ConfusionMatrixDisplay.from_estimator(best_clf, x_test_transformed, y_test, cmap=plt.cm.Blues)
```
![Confusion Matrix](https://github.com/MehranZdi/horse_survival/blob/main/confusion_matrix.png)
