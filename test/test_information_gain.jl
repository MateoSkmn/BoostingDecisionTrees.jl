@testset "information_gain" begin
    # Example numeric dataset
    X = [
        2.0  3.0;
        1.0  2.0;
        4.0  3.0;
        3.0  1.0;
        5.0  2.0
    ]

    # Corresponding class labels
    y = ["No", "No", "Yes", "Yes", "Yes"]

    # Test information gain for each feature
    threshold1, gain1 = information_gain(X[:, 1], y)
    threshold2, gain2 = information_gain(X[:, 2], y)

    println("Feature 1: best threshold = ", threshold1, ", gain = ", gain1)
    println("Feature 2: best threshold = ", threshold2, ", gain = ", gain2)

    # Pick the feature with the maximum gain
    if gain1 > gain2
        best_feature = 1
        best_threshold = threshold1
        best_gain = gain1
    else
        best_feature = 2
        best_threshold = threshold2
        best_gain = gain2
    end

    println("Best feature index: ", best_feature)
    println("Best threshold: ", best_threshold)
    println("Best information gain: ", best_gain)

    @test best_feature == 1          
    @test best_gain > 0.0            
    @test best_threshold > 0.0       
end
