# BoostingDecisionTrees.jl
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MateoSkmn.github.io/BoostingDecisionTrees.jl/dev/)
[![Build Status](https://github.com/MateoSkmn/BoostingDecisionTrees.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/MateoSkmn/BoostingDecisionTrees.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/MateoSkmn/BoostingDecisionTrees.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MateoSkmn/BoostingDecisionTrees.jl)

This project demonstrates how different tree based models handle multiclass classification using the Iris Dataset. By comparing a single split (stump) to a full tree and an adaptive ensemble (AdaBoost), we can show the "power" of each approach.

# Running the package
To use this package, open a Julia REPL and run:
```shell
pkg> add https://github.com/MateoSkmn/BoostingDecisionTrees.jl

julia> using BoostingDecisionTrees
```


# Examples
### Load dataset
For easier use you may download the given dataset under 'src/data/Iris.csv'. The dataset can also be found at https://www.kaggle.com/datasets/uciml/iris.
```shell
julia>  X,y = load_data("path/to/Iris.csv")
```
This loads the 150 samples in a random order.
**Attention:** this method was created especially for the described Iris.csv and might not work for other datasets.
You can always use your own dataset as long as X is a Matrix and y is a Vector.

### Decision Stump
When you have defined a dataset you can start training your classifiers. The most simple one is the decision stump.
You can train a stump with the created dataset and predict labels for your dataset. The following example shows you how to use a simple training and test split
```shell
julia> stump = train_stump(X[1:100, :], y[1:100])

julia> prediciton = predict_stump(stump, X[101:150, :])

julia> sum(prediciton == y[101:150]) / size(y[101:150], 1) # Accuracy of the created model
```

### Decision Tree
```shell
julia> ada = train_tree(X[1:100, :], y[1:100]; max_depth=5)

julia> ada2 = train_tree(X[1:100, :], y[1:100]) # This will use the same parameters as the code above

julia> prediciton = predict_tree(ada, X[101:150, :])

julia> sum(prediciton == y[101:150]) / size(y[101:150], 1) # Accuracy of the created model
```
**TODO**

### AdaBoost
AdaBoost is an ensemble learning classifier using multiple weaker learners. Each new learner focuses on correcting the errors made by its predicessors.
You can train a model using your dataset. You may also adjust the maximum number of iterations as well as the maximum 'power' of a weaker learner.
```shell
julia> ada = train_adaboost(X[1:100, :], y[1:100]; iterations=50, max_alpha=2.5)

julia> ada2 = train_adaboost(X[1:100, :], y[1:100]) # This will use the same parameters as the code above

julia> prediciton = predict(ada, X[101:150, :])

julia> sum(prediciton == y[101:150]) / size(y[101:150], 1) # Accuracy of the created model
```

### Further
**TODO**: In later development you will be able to switch between the splitting criteria 'gini impurity' and 'information gain'.
As for now only 'gini impurity' will be used when creating models.