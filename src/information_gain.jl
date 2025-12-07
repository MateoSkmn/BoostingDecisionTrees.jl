# --- Code for Information Gain --- #

""" This is needed to later calculate IG """
function entropy(y::Vector)
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
    Computes the information gain obtained by splitting on one feature column. 
    This calculates how helpful a selector is by comparing entropy of a feature before and after applying it
"""
function information_gain(X_column::Vector, y::Vector)
    # Compute entropy BEFORE the split (parent node)
    parent_entropy = entropy(y)

    # Total number of samples
    n = length(y)

    # Find all unique values in this feature
    feature_values = unique(X_column)

    # Initialize weighted entropy after the split
    weighted_entropy = 0.0

    # Iterate over each unique feature value
    for v in feature_values
        # Find indices where feature equals v
        indices = findall(x -> x == v, X_column)

        # Extract corresponding labels for this split
        y_subset = y[indices]

        # Weight = proportion of samples in this subset
        weight = length(y_subset) / n

        # Add weighted entropy for this subset
        weighted_entropy += weight * entropy(y_subset)
    end

    # Information Gain = Parent Entropy - Weighted Child Entropy
    return parent_entropy - weighted_entropy
end

""" Finds the feature index that gives the highest information gain. """
function best_split_information_gain(X::Matrix, y::Vector)
    # Number of features (columns)
    num_features = size(X, 2)

    # Store the best gain found
    best_gain = -Inf

    # Store index of best feature
    best_feature = -1

    # Loop over each feature column
    for feature_idx in 1:num_features
        # Extract column as vector
        X_column = X[:, feature_idx]

        # Compute information gain for this feature
        gain = information_gain(X_column, y)

        # Update best feature if this gain is higher
        if gain > best_gain
            best_gain = gain
            best_feature = feature_idx
        end
    end

    return best_feature, best_gain
end
