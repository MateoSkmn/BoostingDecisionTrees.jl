"""
BoostingDecisionTrees

A Julia module for decision tree and boosting algorithms, including decision stumps, Gini impurity,
and information gain utilities.

# Overview
This module provides tools for training and evaluating simple decision trees,
with support for both Gini impurity and information gain as splitting criteria. 

# Features
- *Splitting Criteria*: Supports both Gini impurity and information gain for feature selection.
- *Utilities*: Includes helper functions for entropy, Gini impurity, and majority voting.

# Exports
- *Decision TreeNode Functions*:
  - train_tree: Train a decision tree on a dataset.
  - predict: Make predictions using a trained decision tree.
- *AdaBoost Functions*
  - train_adaboost: Train an AdaBoost model on a dataset.
  - predict: Make predicitions using a trained AdaBoost model

"""
module BoostingDecisionTrees

using StatsBase: countmap, sample, Weights
using CSV: read
using DataFrames: DataFrame, nrow
using Random: shuffle

include("gini_impurity.jl")
include("information_gain.jl")
include("DecisionTree.jl")
include("data_loader.jl")
include("AdaBoost.jl")

export
    TreeNode,
    DecisionNode,
    LeafNode,
    train_tree,
    predict,
    load_data_iris,
    AdaBoost,
    train_adaboost,
    information_gain,
    best_split

end