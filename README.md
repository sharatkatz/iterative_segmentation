Iterative Segmentation
==

### Build
Build wheel using
```
python setup.py sdist bdist_wheel
```

### Install
```
pip install --upgrade pip
pip install packagename
```

### Code files

#### 1. iterative_segmentation.py
* Implements Class `InputRecoding`, prepares data ready to do clustering
    * Imports input csv file as pandas dataframe.
    * Converts the flag columns to boolean form.
    * Converts the categorical columns into binary dummies each representing
    the presence or absence of a certain member in a category.
    * Excludes the variables which are not used for clustering e.g. entity\_id etc.
* Implements Class `ImputeSteps`
    * Prepares the missing values report.
    * Imputes selected columns with a zero (for now).
* Implementsclass `Segments`
    * Options of clustering methods provided are `kmeans`, `agglomerative`, `fuzzy-c-means`, `birch`, `dbscan`, `affinitypropagation`, `spectralclustering`, `gaussianmixture` and `optics`.
    Refer [scikit clustering methods](https://scikit-learn.org/stable/modules/clustering.html) for information on algorithms.
    * Each of the clustering method is implemented as:
        * First attempt of clustering is carried out on the entire input data.
        * If some of the clusters obtained in the first attempt have members < 2pct of the input data
        then all the members of those clusters are mixed and sent to second attempt of clustering with the
        same algorithm. And the number of clusters requested in the subsequent step are 2 less than that requested in the prior step.
        * Above process is repeated till we obtain at least 6 clusters in a step, so that we end up with a relatively healthier number of members per cluster eventually.
        * This way of incremental clustering is obtained using recursive implementation instead of explicit looping as the number of steps required to reach to acceptance criteria are unknown at the beginning.
        * Since, by default the algorithm assign int nums to clusters found in a certain step, to distinguish between clusters found recursively an alphabet letter is used as prefix. `A_clusternum` for first set of clusters, `B_clusternum` for the second set of clusters and so on.
        * This class also implements producing silhoutte plots with scores

#### 2. explainer.py
* Implements class `ModelPipeline`
    * Once, clusters are obtained an xgboost supervised classifier is implemented to do cluster prediction with the aid of full random/grid search over parameters to train and evaluate the model.
    Refer [Python API Reference](https://xgboost.readthedocs.io/en/latest/python/python_api.html) for more information on xgboost algorithm and implementation in Python.
    * After model building explanatory variables are plotted based on the feature importance metrics and also plotted is shap summary using shap.TreeExplainer. 
    Refer [shap API reference](https://shap-lrjball.readthedocs.io/en/docs_update/generated/shap.TreeExplainer.html) for information on shap implementation in Python.