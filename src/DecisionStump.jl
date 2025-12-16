using StatsBase: countmap

"""
    majority_label(y)

Return the most common class label in `y`.

# Arguments
- `y::AbstractVector`: a collection of class labels.

# Returns
- The label that appears most frequently in `y`. If there is a tie, one of the tied labels is returned.
"""
function majority_label(y)
    counts = countmap(y)
    max_label = nothing
    max_count = -Inf

    for (label, c) in counts
        if c > max_count
            max_count = c
            max_label = label
        end
    end

    return max_label
end


"""
    DecisionStump

A simple binary classifier based on a single feature threshold.

# Fields
- `feature::Int`: Index of the feature to split on.
- `threshold::Float64`: Threshold value for the split.
- `left_label`: Class label for samples where feature value <= threshold.
- `right_label`: Class label for samples where feature value > threshold.

# Examples
```jldoctest
julia> stump = DecisionStump(1, 2.5, "A", "B")
DecisionStump(1, 2.5, "A", "B")
```
"""
struct DecisionStump
    feature::Int
    threshold::Float64
    left_label
    right_label
end

"""
    train_stump(X, y)

Train a decision stump classifier on the dataset.

# Arguments
- `X::AbstractMatrix`: rows are samples, columns are features.
- `y::AbstractVector`: class labels for each sample.

# Returns
- `DecisionStump`: a trained stump with `feature`, `threshold`, `left_label`, and `right_label`.

# Examples
```jldoctest
julia> X = [1.0 2.0; 3.0 0.5; 2.0 1.5]
julia> y = ["a", "b", "a"]
julia> stump = train_stump(X, y)
DecisionStump(...)
```
"""
function train_stump(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)

    best_feature = 0
    best_threshold = 0.0
    best_gini = Inf

    for j in 1:n_features 
        feature_col = X[:, j] 

        threshold, gini = best_split(feature_col, y)

        if threshold === nothing
            continue
        end

        if gini < best_gini
            best_gini = gini
            best_threshold = threshold
            best_feature = j
        end
    end

    # If no split is possible
    if best_feature == 0
        lbl = majority_label(y)
        return DecisionStump(1, -Inf, lbl, lbl)
    end

    feature_col = X[:, best_feature]
    left_idx  = findall(x -> x <= best_threshold, feature_col)
    right_idx = findall(x -> x >  best_threshold, feature_col)

    left_label  = majority_label(y[left_idx])
    right_label = majority_label(y[right_idx])

    return DecisionStump(best_feature, best_threshold, left_label, right_label)
end


"""
    predict_stump(stump, X)

Make predictions using the trained decision stump.

# Arguments
- 'stump::DecisionStump': a trained decision stump model
- 'X'::AbstractMatrix': rows are samples, columns are features

# Returns
- `Vector{Any}`: predicted class labels for each sample in `X`

# Examples
```jldoctest
julia> stump = DecisionStump(1, 2.5, "A", "B")
julia> X = [1.0 2.0; 3.0 0.5; 2.0 1.5]
julia> preds = predict_stump(stump, X)
["A", "B", "A"]
```
"""
function predict_stump(stump::DecisionStump, X::AbstractMatrix)
    n = size(X, 1)
    preds = Vector{Any}(undef, n)

    for i in 1:n
        if X[i, stump.feature] <= stump.threshold
            preds[i] = stump.left_label
        else
            preds[i] = stump.right_label
        end
    end

    return preds
end
