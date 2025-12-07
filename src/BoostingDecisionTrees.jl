module BoostingDecisionTrees

using StatsBase: countmap

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
end
