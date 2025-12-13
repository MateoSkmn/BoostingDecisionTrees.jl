using BoostingDecisionTrees
using Test

@testset "BoostingDecisionTrees.jl" begin
    # Write your tests here.
    include("test_decision_stump.jl")
    include("test_gini_impurity.jl")
end
