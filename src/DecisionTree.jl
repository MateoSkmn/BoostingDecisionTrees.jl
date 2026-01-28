#TODO: AI Note
# -------------------- Utility Functions --------------------

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

# -------------------- Node Types --------------------

"""
    TreeNode

Abstract type for decision tree nodes.
"""
abstract type TreeNode end

"""
    DecisionNode

A decision node in a decision tree that splits data based on a feature and threshold.

# Fields
- `feature::Int`: the feature index used for splitting.
- `threshold::Float64`: the threshold value for the split.
- `left::TreeNode`: the left subtree (samples where feature ≤ threshold).
- `right::TreeNode`: the right subtree (samples where feature > threshold).
"""
struct DecisionNode <: TreeNode
    feature::Int
    threshold::Float64
    left::TreeNode
    right::TreeNode
end

"""
    LeafNode

A leaf node holding a predicted class label.
"""
struct LeafNode <: TreeNode
    label
end

# -------------------- Training --------------------

"""
    train_tree(X, y; max_depth=5, criterion=:gini)

Train a decision tree using numeric threshold splits.

# Arguments
- `X::AbstractMatrix`: feature matrix.
- `y::AbstractVector`: class labels.
- `max_depth::Int`: maximum tree depth.
- `criterion::Symbol`: `:information_gain` or `:gini`.

# Returns
- `TreeNode`
"""
function train_tree(X::AbstractMatrix, y::AbstractVector;
                    max_depth::Int=5,
                    criterion::Symbol=:gini)

    n_samples, n_features = size(X)

    # Stopping conditions
    if n_samples == 0 || length(unique(y)) == 1 || max_depth <= 0
        return LeafNode(majority_label(y))
    end

    best_feature = 0
    best_threshold = 0.0

    if criterion == :information_gain
        best_score = -Inf

        for j in 1:n_features
            threshold, gain = information_gain(X[:, j], y)
            if gain > best_score
                best_score = gain
                best_threshold = threshold
                best_feature = j
            end
        end

        if best_score <= 0
            return LeafNode(majority_label(y))
        end

    elseif criterion == :gini
        best_score = Inf

        for j in 1:n_features
            threshold, gini = best_split(X[:, j], y)  # assumed existing
            if gini < best_score
                best_score = gini
                best_threshold = threshold
                best_feature = j
            end
        end

        if best_feature == 0
            return LeafNode(majority_label(y))
        end

    else
        throw(ArgumentError("Unknown criterion: $criterion"))
    end

    left_idx  = findall(x -> x <= best_threshold, X[:, best_feature])
    right_idx = findall(x -> x > best_threshold, X[:, best_feature])

    left_subtree = train_tree(X[left_idx, :], y[left_idx],
                              max_depth=max_depth - 1,
                              criterion=criterion)

    right_subtree = train_tree(X[right_idx, :], y[right_idx],
                               max_depth=max_depth - 1,
                               criterion=criterion)

    return DecisionNode(best_feature, best_threshold,
                        left_subtree, right_subtree)
end

# -------------------- Prediction --------------------

"""
    predict(node::TreeNode, x)

Predict the class label for a single sample.
"""
function predict(node::TreeNode, x::AbstractVector)
    if node isa LeafNode
        return node.label
    else
        if x[node.feature] <= node.threshold
            return predict(node.left, x)
        else
            return predict(node.right, x)
        end
    end
end

"""
    predict(tree::TreeNode, X::AbstractMatrix)

Predict class labels for multiple samples.
"""
function predict(tree::TreeNode, X::AbstractMatrix)
    n = size(X, 1)
    preds = Vector{Any}(undef, n)
    for i in 1:n
        preds[i] = predict(tree, X[i, :])
    end
    return preds
end
