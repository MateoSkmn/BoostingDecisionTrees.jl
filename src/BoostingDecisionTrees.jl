"""
BoostingDecisionTrees

A Julia module for decision tree and boosting algorithms, including decision stumps, Gini impurity,
and information gain utilities.

# Overview
This module provides tools for training and evaluating simple decision trees and decision stumps,
with support for both Gini impurity and information gain as splitting criteria. It is designed for
educational purposes and prototyping machine learning models.

# Features
- *Splitting Criteria*: Supports both Gini impurity and information gain for feature selection.
- *Utilities*: Includes helper functions for entropy, Gini impurity, and majority voting.

# Exports
- *Decision TreeNode Functions*:
  - train_tree: Train a decision tree on a dataset.
  - predict: Make predictions using a trained decision tree.

# Notes
Intended as a compact educational toolkit and a foundation for experiments with boosting and small
decision-tree learners.
"""
module BoostingDecisionTrees

using StatsBase: countmap
using CSV
using DataFrames
using Random

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
    train_adaboost

end