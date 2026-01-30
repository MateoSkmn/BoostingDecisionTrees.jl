# AI Note:
# Parts of the in-file documentation such as docstrings and code comments are based on AI suggestions
# AI was used to suggest performance and structural improvements and suggestions were taken into account when finalizing the module

# -------------------- Utility Functions --------------------

"""
    majority_label(y::Vector{Int})
    majority_label(y::Vector{String})

Return the most common class label in `y`.

# Arguments
- `y::Vector{Int}`: a vector of integer class labels.
- `y::Vector{String}`: a vector of string class labels (expected format: "label_<index>").

# Returns
- For `Vector{Int}`: the integer label that appears most frequently.
- For `Vector{String}`: the string label "label_<index>" that appears most frequently.
"""
#NOTE: Rewrote majority_label functions with suggestions from an LLM for better performance
# based on profiler output.
function majority_label(y::Vector{T}) where T
    # Create a dictionary for counting (works for all hashable types T)
    counts = Dict{T, Int}()
    sizehint!(counts, length(unique(y)))  # Preallocate memory
    
    # Count the labels (with @inbounds for performance)
    @inbounds for label in y
        counts[label] = get(counts, label, 0) + 1
    end

    # Find the label with the highest frequency
    max_label = first(y)  # Default fallback
    max_count = -1
    for (label, count) in counts
        if count > max_count
            max_count = count
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
function train_tree(X::AbstractMatrix, y::AbstractVector; max_depth::Int=5,criterion::Symbol=:gini)

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

Make a prediction for a single sample `x` using the decision tree.

# Arguments
- `node::TreeNode`: a node in the decision tree.
- `x::AbstractVector`: a single sample.

# Returns
- The predicted class label.

# Examples
```jldoctest
julia> leaf = LeafNode("a");

julia> predict(leaf, [1.0, 2.0])
"a"
```
"""
# NOTE: Using @inline here for performance and restructuring code to traverse
# the tree using a while loop, was a suggestion made by an LLM in response
# to the profiler output. 
@inline function predict(node::TreeNode, x::AbstractVector)
    current_node = node
    while true
        if current_node isa LeafNode
            return current_node.label
        else
            if x[current_node.feature] <= current_node.threshold
                current_node = current_node.left
            else
                current_node = current_node.right
            end
        end
    end
end


"""
    predict(tree::TreeNode, X::AbstractMatrix)

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

julia> preds = predict(tree, X)
3-element Vector{Any}:
 "a"
 "b"
 "a"
```
"""
function predict(tree::TreeNode, X::AbstractMatrix)
    n = size(X, 1)
    # extract type information from initial function call
    first_prediction = predict(tree, X[1, :])
    T = typeof(first_prediction)

    preds = Vector{T}(undef, n)
    preds[1] = first_prediction

    # NOTE: Using @inbounds here for performance, was a suggestion made 
    # by an LLM in response to the profiler output. 
    @inbounds for i in 2:n
        preds[i] = predict(tree, X[i, :])
    end
    return preds
end
