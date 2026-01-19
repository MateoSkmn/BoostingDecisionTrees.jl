"""
BoostingDecisionTrees

A Julia module for decision tree and boosting algorithms, including decision stumps, Gini impurity,
and information gain utilities.

# Overview
This module provides tools for training and evaluating simple decision trees and decision stumps,
with support for both Gini impurity and information gain as splitting criteria. It is designed for
educational purposes and prototyping machine learning models.

# Features
- **Decision Stumps**: Simple binary classifiers based on a single feature threshold.
- **Splitting Criteria**: Supports both Gini impurity and information gain for feature selection.
- **Utilities**: Includes helper functions for entropy, Gini impurity, and majority voting.

# Exports
- **Decision Stump Functions**:
  - `DecisionStump`: A type representing a decision stump.
  - `train_stump`: Train a decision stump on a dataset.
  - `predict_stump`: Make predictions using a trained decision stump.

- **Gini Impurity**:
  - `best_split`: Find the best threshold to split a feature vector using Gini impurity.

- **Information Gain**:
  - `best_split_information_gain`: Find the feature index with the highest information gain.

# Examples
```jldoctest
julia> using BoostingDecisionTrees

julia> X = [1.0 2.0; 3.0 0.5; 2.0 1.5];

julia> y = ["a", "b", "a"];

julia> stump = train_stump(X, y)
DecisionStump(1, 2.5, "a", "b")

julia> predict_stump(stump, X)
3-element Vector{Any}:
 "a"
 "b"
 "a"
```

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
export best_split

include("information_gain.jl")
export best_split_information_gain
# TODO: later when this criterion is used inside the DecisionTree consider renaming
# OK for now as it is not used

include("DecisionStump.jl")

export
    DecisionStump,
    train_stump,
    predict_stump

include("DecisionTree.jl")
    
export
    TreeNode,
    DecisionNode,
    LeafNode,
    train_tree,
    predict_tree

include("data_loader.jl")
export load_data


include("AdaBoost.jl")
export
  AdaBoost,
  train_adaboost,
  createWeightedDataset,
  predict

end