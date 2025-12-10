module BoostingDecisionTrees

using StatsBase: countmap

# Write your package code here.
include("gini_impurity.jl")
export best_split

include("DecisionStump.jl")

export
    DecisionStump,
    train_stump,
    predict_stump
end
