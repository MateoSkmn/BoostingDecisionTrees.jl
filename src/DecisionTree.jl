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
    TreeNode

Abstract type representing a node in a decision tree.

All nodes in a decision tree are subtypes of `TreeNode`.
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

A leaf node in a decision tree that holds a class label for prediction.

# Fields
- `label`: the predicted class label for samples reaching this leaf node.
"""
struct LeafNode <: TreeNode
    label
end

"""
    train_tree(X, y; max_depth=5)

Train a decision tree classifier on the dataset.

# Arguments
- `X::AbstractMatrix`: rows are samples, columns are features.
- `y::AbstractVector`: class labels for each sample.
- `max_depth::Int`: maximum depth of the tree (default: 5).

# Returns
- `TreeNode`: a trained decision tree.

# Examples
```jldoctest
julia> X = [1.0 2.0; 3.0 0.5; 2.0 1.5; 4.0 3.0];

julia> y = ["a", "b", "a", "b"];

julia> tree = train_tree(X, y; max_depth=2)
DecisionNode(1, 2.5, LeafNode("a"), LeafNode("b"))
```
"""
function train_tree(X::AbstractMatrix, y::AbstractVector; max_depth=5)
    n_samples, n_features = size(X)
    unique_labels = unique(y)

    # Base cases:
    # 1. If only one class, return a leaf node.
    # 2. If max_depth is reached, return a leaf node with the majority label.
    if length(unique_labels) == 1 || max_depth <= 0
        return LeafNode(majority_label(y))
    end

    # Find the best split
    best_feature = 0
    best_threshold = 0.0
    best_gini = Inf

    for j in 1:n_features
        feature_col = X[:, j]
        threshold, gini = best_split(feature_col, y)

        if gini < best_gini
            best_gini = gini
            best_threshold = threshold
            best_feature = j
        end
    end

    # Split the data
    feature_col = X[:, best_feature]
    left_idx = findall(x -> x <= best_threshold, feature_col)
    right_idx = findall(x -> x > best_threshold, feature_col)

    # Recursively train left and right subtrees
    left_subtree = train_tree(X[left_idx, :], y[left_idx], max_depth=max_depth-1)
    right_subtree = train_tree(X[right_idx, :], y[right_idx], max_depth=max_depth-1)

    return DecisionNode(best_feature, best_threshold, left_subtree, right_subtree)
end

"""
    predict_tree(node::TreeNode, x)

Make a prediction for a single sample `x` using the decision tree.

# Arguments
- `node::TreeNode`: a node in the decision tree.
- `x::AbstractVector`: a single sample.

# Returns
- The predicted class label.

# Examples
```jldoctest
julia> leaf = LeafNode("a");

julia> predict_tree(leaf, [1.0, 2.0])
"a"
```
"""
function predict_tree(node::TreeNode, x::AbstractVector)
    if node isa LeafNode
        return node.label
    else
        if x[node.feature] <= node.threshold
            return predict_tree(node.left, x)
        else
            return predict_tree(node.right, x)
        end
    end
end

"""
    predict_tree(tree::TreeNode, X::AbstractMatrix)

Make predictions for multiple samples using the decision tree.

# Arguments
- `tree::TreeNode`: a trained decision tree.
- `X::AbstractMatrix`: rows are samples, columns are features.

# Returns
- `Vector{Any}`: predicted class labels for each sample in `X`.

# Examples
```jldoctest
julia> tree = DecisionNode(1, 2.5, LeafNode("a"), LeafNode("b"));

julia> X = [1.0 2.0; 3.0 0.5; 2.0 1.5];

julia> preds = predict_tree(tree, X)
3-element Vector{Any}:
 "a"
 "b"
 "a"
```
"""
function predict_tree(tree::TreeNode, X::AbstractMatrix)
    n = size(X, 1)
    preds = Vector{Any}(undef, n)

    for i in 1:n
        preds[i] = predict_tree(tree, X[i, :])
    end

    return preds
end


