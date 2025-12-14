@testset "best_split" begin
    # Example dataset (categorical features)
    X = [
        "Sunny" "Hot";
        "Sunny" "Hot";
        "Overcast" "Hot";
        "Rain" "Mild";
        "Rain" "Cool"
    ]

    # Corresponding class labels
    y = ["No", "No", "Yes", "Yes", "Yes"]

    # Find best feature to split on
    best_feature, best_gain = best_split_information_gain(X, y)

    println("Best feature index: ", best_feature)
    println("Best information gain: ", best_gain)

    @test best_feature == 1
    @test best_gain == 0.9709505944546686
end