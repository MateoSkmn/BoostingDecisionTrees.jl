using StatsBase: countmap

# Majority label helper 
function majority_label(y)
    counts = countmap(y)
    labels = collect(keys(counts))
    return labels[argmax(values(counts))]
end


# Decision stump struct
struct DecisionStump
    feature::Int
    threshold::Float64
    left_label
    right_label
end

# Train a decision stump
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


# Prediction with the stump
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
