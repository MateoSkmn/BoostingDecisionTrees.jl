module BoostingDecisionTrees

# Write your package code here.
  
using StatsBase: countmap

include("DecisionStump.jl")

export
    DecisionStump,
    train_stump,
    predict_stump
end
