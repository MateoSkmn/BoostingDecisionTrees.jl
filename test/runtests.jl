using BoostingDecisionTrees
using Test

@testset "BoostingDecisionTrees.jl" begin
    include("test_decision_tree.jl")
    include("test_gini_impurity.jl")
    include("test_information_gain.jl")
    include("test_data_loader.jl")
    include("test_ada_boost.jl")
end
