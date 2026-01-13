# Concept and structure adapted from: http://youtube.com/watch?v=LsK-xG1cLYA

"""
    AdaBoost(stumps, alphas)

    A stronger ensemble learning classifier consisting of multiple weaker learners. 
    Each new learner focusess on correcting the errors made by its predicessors.

    # Fields
    - `stumps::Vector{DecisionStump}`: A collection of DecisionStump objects. Each stump acts as a weak classifier that makes a prediction based on a single feature threshold.
    - `alphas::Vector{Float64}`: A vector of floating-point weights, corresponding to the voting power of a stumps. A higher alpha values means the stump was more accurate during the training phase.
"""
struct AdaBoost
    stumps::Vector{DecisionStump}
    alphas::Vector{Float64}
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

    # Returns
    - `AdaBoost`: a trained classifier with `stumps` and `alphas`.
"""
function train_adaboost(X::AbstractMatrix, y::AbstractVector; iterations::Integer = 50, max_alpha::Float64 = 2.5)
    if iterations < 1
        throw(ArgumentError("iterations must be at least 1" ))
    end

    n = size(X, 1)
    w = fill(1.0 / n, n)

    stumps = DecisionStump[]
    alphas = Float64[]

    for i in 1:iterations
        # Train a DecisionStump
        stump = train_stump(X, y)
        # Get total error of the DecisionStump
        stump_prediction = predict_stump(stump, X)
        err = sum(stump_prediction .!= y) / size(y, 1)
        # Calculate alpha (Amount of say)
        # log(0) => -Inf || 1/0 => Inf
        alpha = 0.5 * log((1 - err) / err)

        push!(stumps, stump)
        push!(alphas, min(alpha, max_alpha))
        # alpha can go from -Inf (The stump did everything wrong) to max_alpha (the stump did everything correct)
        # As Inf would cause the stump to become a 'Dictator' making only that stump responsable for a decision

        # Early break because there are no past errors to learn from
        if alpha == Inf
            break
        end

        # Update sample weights
        for i in eachindex(y)
            if stump_prediction[i] != y[i]
                w[i] *= exp(alpha)      # incorrectly classified
            else
                w[i] *= exp(-alpha)     # correctly classified
            end
        end
        # Normalize weights
        w = w ./ sum(w)
        # Create new dataset based on the errors made by the DecisionStump
        X, y = createWeightedDataset(X,y,w)
        # Repeat for the amount of iterations
    end

    return AdaBoost(stumps, alphas)
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
        
        # In case of floating point errors, default to the last index
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

The function adds the weighted votes of all decision stumps
within the model to determine the most likely class for each sample.

### Arguments
- `model::AdaBoost`: A trained AdaBoost structure.
- `X::AbstractMatrix`: rows are samples, columns are features.

### Returns
- `Vector`: A vector of predicted labels, with the same type as the labels found 
  in the model's stumps.
"""
function predict(model::AdaBoost, X::AbstractMatrix)
    n_samples = size(X, 1)
    
    # 1. Dynamically find all unique labels used across the entire ensemble
    all_possible_labels = Set()
    for s in model.stumps
        push!(all_possible_labels, s.left_label)
        push!(all_possible_labels, s.right_label)
    end
    labels = collect(all_possible_labels)

    # 2. Initialize scores for each sample and each potential label
    # scores[label] maps to a Vector of weights for each sample
    scores = Dict(lbl => zeros(Float64, n_samples) for lbl in labels)

    # 3. Aggregate weighted votes from every stump
    for (stump, alpha) in zip(model.stumps, model.alphas)
        feature_col = X[:, stump.feature]
        
        for i in 1:n_samples
            # Decide label based on threshold
            pred = feature_col[i] <= stump.threshold ? stump.left_label : stump.right_label
            
            # Add this stump's "voice" (alpha) to the total for that label
            scores[pred][i] += alpha
        end
    end

    # 4. Final Prediction: For each sample, pick the label with the highest score
    final_preds = Vector{eltype(labels)}(undef, n_samples)
    for i in 1:n_samples
        # argmax returns the label that maximizes the score at index i
        final_preds[i] = argmax(lbl -> scores[lbl][i], labels)
    end

    return final_preds
end

"""
    predict(model::AdaBoost, X::AbstractVector)

A convenience method for predicting the label of a single sample.

### Arguments
- `X::AbstractVector`: A single sample represented as a vector of features.

### Returns
- The predicted label for the single input sample.
"""
function predict(model::AdaBoost, X::AbstractVector)
    return predict(model, reshape(X, 1, :))
end