"""
    gini_impurity(classes)

Compute the Gini impurity of a vector of class labels 'classes'.
"""
function gini_impurity(classes)
    len = length(classes)
    if len == 0
        return 0
    end
    # https://juliastats.org/StatsBase.jl/v0.17/counts.html#StatsBase.countmap 
    # Create a dictionary of the classes (keys) and how often each one appears (values) in the given data
    counts = countmap(classes)
    return 1 - sum((c/len)^2 for c in values(counts))
end

"""
    best_split(feature, labels)

Parameters:
- 'feature': a vector of numerical values
- 'labels': a vector of class labels (same length as 'feature')

Return:
- 'best_threshold': best numerical value to split the 'feature' on
- 'best_gini': weighted gini impurity after the split
"""
function best_split(feature, labels)
    n = length(feature)
    best_gini = Inf
    best_threshold = nothing

    if n == 0
        throw(ArgumentError("best_split received empty feature array"))
    end
    if n != length(labels)
        throw(DimensionMismatch("feature and labels must have same length"))
    end
    if n == 1
        return best_threshold, best_gini   # no possible split with a single point
    end
    # Sort by feature to have consecutive array
    # sortperm(x) returns array of indeces of an ordered x
    order = sortperm(feature)
    f_sorted = feature[order]
    y_sorted = labels[order]

    # Consider unique mid of two feature points as thresholds
    # Simplified version of x[i] + (x[i+1] - x[i])/2 by Carlos Guestrin (2013) p. 18
    # http://courses.cs.washington.edu/courses/cse446/13sp/slides/decision-trees-boosting-annotated.pdf
    thresholds = unique((f_sorted[1:end-1] .+ f_sorted[2:end]) ./ 2)

    for t in thresholds
        # Get all indeces of features that are above/below the current threshold
        # searchsortedlast(a, x) returns last index where a[i] <= x
        split_idx = searchsortedlast(f_sorted, t)
        left_idx = 1:split_idx
        right_idx = (split_idx + 1):n

        # Get labels for the two branches
        y_left  = y_sorted[left_idx]
        y_right = y_sorted[right_idx]

        # Compute weighted gini impurity, to balance out differently sized splits
        # e.g. with two classes: 1 sample (Gini = 0) & 999 samples (worst case Gini = 0.5)
            # without weight: (0 + 0.5)/2 = 0.25 ==> Good result
            # with weight: (1/1000)*0 + (999/1000)*0.5 = 0.4995 ==> Basically worst case
        weight_left = length(y_left)/n
        weight_right = length(y_right)/n
        gini_weighted = weight_left * gini_impurity(y_left) + weight_right * gini_impurity(y_right)

        if gini_weighted < best_gini
            best_gini = gini_weighted
            best_threshold = t
        end

        # No need to go further if a perfect gini impurity was found
        if best_gini == 0
            break
        end
    end

    return best_threshold, best_gini
end
