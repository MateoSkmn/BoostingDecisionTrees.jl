# AI Note:
# Parts of the in-file documentation such as docstrings and code comments are based on AI suggestions
# AI was used to suggest performance and structural improvements and suggestions were taken into account when finalizing the module

"""
    entropy(y::AbstractVector)

Compute the entropy of a AbstractVector of class labels.

# Arguments
- `y::AbstractVector`: A AbstractVector of class labels.

# Returns
- `Float64`: The entropy of the input AbstractVector.
"""

function entropy(y::AbstractVector)
    # Total number of samples
    n = length(y)

    # Count unique classes
    class_counts = Dict{Any, Int}()
    for label in y
        class_counts[label] = get(class_counts, label, 0) + 1
    end

    # Initialize entropy value
    H = 0.0

    # Loop over each class and compute entropy contribution
    for (_, count) in class_counts
        # Probability of this class
        p = count / n

        # Entropy formula: -p * log2(p)
        H -= p * log2(p)
    end

    return H
end

"""
    information_gain(X_column::AbstractVector{<:Real}, y::AbstractVector)

Compute the best information gain obtainable by splitting a numeric feature column
using a threshold (x ≤ t vs. x > t).

# Returns
- `best_threshold::Float64`: threshold yielding maximum information gain
- `best_gain::Float64`: corresponding information gain
"""
function information_gain(X_column::AbstractVector{<:Real}, y::AbstractVector)

    n = length(y)
    n == 0 && return (0.0, 0.0)

    parent_entropy = entropy(y)

    # Sort unique feature values
    values = sort(unique(X_column))

    # No split possible if only one unique value
    length(values) ≤ 1 && return (0.0, 0.0)

    best_gain = -Inf
    best_threshold = values[1]  # default (won't be used if gain ≤ 0)

    # Candidate thresholds: midpoints between consecutive values
    for i in 1:length(values)-1
        t = (values[i] + values[i + 1]) / 2

        left_idx  = findall(x -> x ≤ t, X_column)
        right_idx = findall(x -> x > t, X_column)

        # Skip invalid splits
        if isempty(left_idx) || isempty(right_idx)
            continue
        end

        y_left  = y[left_idx]
        y_right = y[right_idx]

        weighted_entropy =
            (length(y_left)  / n) * entropy(y_left) + (length(y_right) / n) * entropy(y_right)

        gain = parent_entropy - weighted_entropy

        if gain > best_gain
            best_gain = gain
            best_threshold = t
        end
    end

    return best_threshold, best_gain
end
