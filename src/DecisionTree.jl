using StatsBase: countmap

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

Abstract type representing a node in a decision tree.
"""
abstract type TreeNode end

"""
    DecisionNode

A decision node in a decision tree.

# Fields
- `feature::Int`: The feature index used for splitting.
- `threshold::Float64`: Numeric threshold for split (NaN if categorical).
- `children::Union{Dict{Any, TreeNode}, Nothing}`: Mapping of categorical feature values to subtrees.
- `right::Union{TreeNode, Nothing}`: Right subtree for numeric splits.
"""
struct DecisionNode <: TreeNode
    feature::Int
    threshold::Float64
    children::Union{Dict{Any, TreeNode}, Nothing}  # for categorical splits
    left::Union{TreeNode, Nothing}                 # left subtree for numeric splits
    right::Union{TreeNode, Nothing}                # right subtree for numeric splits
end

"""
    LeafNode

A leaf node in a decision tree that holds a class label for prediction.

# Fields
- `label`: the predicted class label.
"""
struct LeafNode <: TreeNode
    label
end

# -------------------- Unified Training Function --------------------

"""
    train_tree(X, y; max_depth=5, criterion=:gini)

Train a decision tree using the specified criterion.

# Arguments
- `X::Matrix`: Feature matrix.
- `y::Vector`: Class labels.
- `max_depth::Int`: Maximum tree depth.
- `criterion::Symbol`: `:gini` for numeric, `:information_gain` for categorical.

# Returns
- `TreeNode`: Trained decision tree.
"""
function train_tree(X::AbstractMatrix, y::AbstractVector;
                    max_depth::Int=5, criterion::Symbol=:gini)
    n_samples, n_features = size(X)
    unique_labels = unique(y)

    # Base case: pure node or max depth reached
    if length(unique_labels) == 1 || max_depth <= 0
        return LeafNode(majority_label(y))
    end

    if criterion == :gini
        # numeric Gini split
        best_feature = 0
        best_threshold = 0.0
        best_gini = Inf

        for j in 1:n_features
            threshold, gini = best_split(X[:, j], y)
            if gini < best_gini
                best_gini = gini
                best_threshold = threshold
                best_feature = j
            end
        end

        left_idx = findall(x -> x <= best_threshold, X[:, best_feature])
        right_idx = findall(x -> x > best_threshold, X[:, best_feature])

        left_subtree = train_tree(X[left_idx, :], y[left_idx],
                                  max_depth=max_depth-1, criterion=:gini)
        right_subtree = train_tree(X[right_idx, :], y[right_idx],
                                   max_depth=max_depth-1, criterion=:gini)

        return DecisionNode(best_feature, best_threshold, nothing, left_subtree, right_subtree)

    elseif criterion == :information_gain
        # categorical split
        best_feature, best_gain = best_split_information_gain(X, y)
        if best_gain <= 0
            return LeafNode(majority_label(y))
        end

        children = Dict{Any, TreeNode}()
        for val in unique(X[:, best_feature])
            idx = findall(x -> x == val, X[:, best_feature])
            X_subset = X[idx, :]
            y_subset = y[idx]
            children[val] = train_tree(X_subset, y_subset,
                                       max_depth=max_depth-1, criterion=:information_gain)
        end

        return DecisionNode(best_feature, NaN, children, nothing, nothing)

    else
        throw(ArgumentError("Unknown criterion: $criterion"))
    end
end

# -------------------- Prediction Function --------------------

"""
    predict_tree(node::TreeNode, x::AbstractVector)

Predict a single sample using a decision tree (numeric or categorical splits).
"""

function predict_tree(node::TreeNode, x::AbstractVector)
    if node isa LeafNode
        return node.label
    else
        if !isnan(node.threshold)  # numeric
            if x[node.feature] <= node.threshold
                return predict_tree(node.left, x)
            else
                return predict_tree(node.right, x)
            end
        else  # categorical
            val = x[node.feature]
            if haskey(node.children, val)
                return predict_tree(node.children[val], x)
            else
                # fallback: majority label among children
                child_preds = [predict_tree(c, x) for c in values(node.children)]
                return majority_label(child_preds)
            end
        end
    end
end


"""
    predict_tree(tree::TreeNode, X::AbstractMatrix)

Predict multiple samples using a decision tree.
"""
function predict_tree(tree::TreeNode, X::AbstractMatrix)
    n = size(X, 1)
    preds = Vector{Any}(undef, n)
    for i in 1:n
        preds[i] = predict_tree(tree, X[i, :])
    end
    return preds
end

