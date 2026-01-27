# Concept and structure adapted from: http://youtube.com/watch?v=LsK-xG1cLYA

"""
    AdaBoost(learner, alphas)

    A stronger ensemble learning classifier consisting of multiple weaker learners. 
    Each new learner focuses on correcting the errors made by its predicessors.

    # Fields
    - `learner::Vector{DecisionTree}`: A collection of DecisionTree objects. Each tree acts as a weak classifier that makes a prediction based on a single feature threshold.
    - `alphas::Vector{Float64}`: A vector of floating-point weights, corresponding to the voting power of a tree. A higher alpha values means the stump was more accurate during the training phase.
"""
struct AdaBoost{T}
    learners::Vector{TreeNode}
    alphas::Vector{Float64}
    labels::Vector{T}
end

# max_alpha = 2.5, happens for an accuracy of >= 99.999%. The bigger the value the more of a 'dictator' becomes a stump with a perfect result
"""
    train_adaboost(X, y; iterations, max_alpha)

    Trains an AdaBoost classifier on the given dataset.

    # Arguments
    - `X::AbstractMatrix`: rows are samples, columns are features.
    - `y::AbstractVector`: class labels for each sample.
    - `iterations::Integer`: maximum number of weak learners. In case of perfect fit the training will be stopped early. Values must be in range [1, Inf). Default is 50.
    - `max_alpha::Float64`: A threshold to cap the "amount of say" (alpha) for any single stump. Default is 2.5 which means an accuracy of >= 99.999%. The bigger the value the more of a 'dictator' becomes a stump with a perfect result.
    - `max_depth::Integer`: Maximum depth of each tree. Default is set to 1 which is equivalent to a decision stump

    # Returns
    - `AdaBoost`: a trained classifier with `learners` and `alphas`.
"""
function train_adaboost(X::AbstractMatrix, y::AbstractVector{T}; iterations::Integer = 50, max_alpha::Float64 = 2.5, max_depth::Integer=1) where T
    if iterations < 1
        throw(ArgumentError("iterations must be at least 1" ))
    end

    n = size(X, 1)
    w = fill(1.0 / n, n)

    learners = TreeNode[]
    alphas = Float64[]
    labels = unique(y)

    for i in 1:iterations
        # Train a DecisionStump
        tree = train_tree(X, y; max_depth=max_depth)
        # Get total error of the DecisionStump
        tree_prediction = predict_tree(tree, X)
        err = sum(tree_prediction .!= y) / size(y, 1)
        # Calculate alpha (Amount of say)
        # log(0) => -Inf || 1/0 => Inf
        alpha = 0.5 * log((1 - err) / err)

        push!(learners, tree)
        push!(alphas, min(alpha, max_alpha))
        # alpha can go from -Inf (The stump did everything wrong) to max_alpha (the stump did everything correct)
        # As Inf would cause the stump to become a 'Dictator' making only that stump responsable for a decision

        # Early break because there are no past errors to learn from
        if alpha == Inf
            break
        end

        # Update sample weights
        for i in eachindex(y)
            if tree_prediction[i] != y[i]
                w[i] *= exp(alpha)      # incorrectly classified
            else
                w[i] *= exp(-alpha)     # correctly classified
            end
        end
        # Normalize weights
        w = w ./ sum(w)
        # Create new dataset based on the errors made by the DecisionTree
        X, y = createWeightedDataset(X,y,w)
        # Repeat for the amount of iterations
    end

    return AdaBoost{T}(learners, alphas, labels)
end

# helper
"""
    createWeightedDataset(X, y, weights)

    Create a new dataset by sampling rows from `X` and `y`, guided by a 
    probability distribution defined by `weights` where samples with higehr weights are more likely to be selected for the new dataset

    # Arguments
    - `X::AbstractMatrix`: rows are samples, columns are features.
    - `y::AbstractVector`: class labels for each sample.
    - `weights::Vector{Float64}`: weight of each sample in the given dataset. The sum of all weights should be 1.

    # Returns
    - `X_prime`: A resampled matrix of the same dimensions and type as `X`.
    - `y_prime`: A resampled vector of the same length and type as `y`.
"""
function createWeightedDataset(X::AbstractMatrix, y::AbstractVector, weights::Vector{Float64})
    n_samples = size(X, 1)
    n_features = size(X, 2)
    
    # Pre-allocate the new X' and y'
    X_prime = similar(X)
    y_prime = similar(y)
    
    # Compute the cumulative sum of weights
    cumulative_weights = cumsum(weights)
    
    # Repeat until X' has the same amount of data as X
    for i in 1:n_samples
        r = rand()
        
        # Find the first index where cumulative weight >= random value
        # findfirst returns the index of the first 'true' element
        idx = findfirst(cw -> cw >= r, cumulative_weights)
        
        # NOTE: This is a fallback and ideally should not happen
        if isnothing(idx)
            idx = n_samples
        end
        
        X_prime[i, :] = X[idx, :]
        y_prime[i] = y[idx]
    end

    return X_prime, y_prime
end

"""
    predict(model, X)

Predict class labels for samples in `X` using a trained AdaBoost classifier.

The function adds the weighted votes of all decision trees
within the model to determine the most likely class for each sample.

### Arguments
- `model::AdaBoost`: A trained AdaBoost structure.
- `X::AbstractMatrix`: rows are samples, columns are features.

### Returns
- `Vector`: A vector of predicted labels, with the same type as the labels found 
  in the model's learners.
"""
function predict(model::AdaBoost, X::AbstractMatrix)
    n_samples = size(X, 1)

    # Set 0 scores for each label the model can predict
    scores = Dict(lbl => zeros(Float64, n_samples) for lbl in model.labels)

    # Sum up votes from every tree
    for (tree, alpha) in zip(model.learners, model.alphas)
        for i in 1:n_samples
            pred = predict_tree(tree, X[i, :])
            scores[pred][i] += alpha
        end
    end

    # argmax returns the label with maximum score at index i
    return [argmax(lbl -> scores[lbl][i], model.labels) for i in 1:n_samples]
end

"""
    predict(model::AdaBoost, X::AbstractVector)

A convenience method for predicting the label of a single sample.

### Arguments
- `model::AdaBoost`: A trained AdaBoost structure.
- `X::AbstractVector`: A single sample represented as a vector of features.

### Returns
- The predicted label for the single input sample.
"""
function predict(model::AdaBoost, X::AbstractVector)
    return predict(model, reshape(X, 1, :))
end