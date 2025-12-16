# BoostingDecisionTrees.jl
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MateoSkmn.github.io/BoostingDecisionTrees.jl/dev/)
[![Build Status](https://github.com/MateoSkmn/BoostingDecisionTrees.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/MateoSkmn/BoostingDecisionTrees.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/MateoSkmn/BoostingDecisionTrees.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MateoSkmn/BoostingDecisionTrees.jl)

# Running the package locally
To use this package locally, open a Julia REPL inside the project structure and run:
```shell
julia> ]activate .

julia> include("src/BoostingDecisionTrees.jl")

julia> using .BoostingDecisionTrees
```


# Examples
### Gini Impurity
First define your dataset that you want to train on.
```shell
julia>  X = [1 2; 2 3; 11 21; 12 22]

julia> y = [0, 0, 1, 1]

julia> stump = train_stump(X, y)
```
Please keep in mind that the current implementation only allows a matrix for X!
You can test with just one feature by reshaping a Vector:
```shell
julia> x = [1,2,3,4]

julia> X = reshape(x, :, 1)

julia> stump = train_stump(X, y)
```

Now you can use the created stump to predict the labels for an input matrix, e.g.:
```shell
julia>  predict_stump(stump, X)
```