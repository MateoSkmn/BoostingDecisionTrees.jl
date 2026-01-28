X = [6.7  2.5  5.8  1.8
    5.7  2.8  4.5  1.3
    6.7  3.0  5.2  2.3
    6.8  2.8  4.8  1.4
    5.9  3.2  4.8  1.8
    6.1  2.6  5.6  1.4
    7.2  3.6  6.1  2.5
    6.1  3.0  4.6  1.4
    4.4  3.0  1.3  0.2
    6.7  3.3  5.7  2.1]
y = [0, 1, 0, 1, 1, 0, 0, 1, 2, 0]

ada_boost = train_adaboost(X,y; iterations=50, max_alpha=2.5)

@testset "AdaBoost training" begin
    @test size(ada_boost.learners, 1) < 50 #Early break, because there is not a lot of data
    @test maximum(ada_boost.alphas) == 2.5 #Final value should always be the max_alpha if number of stumps is less then iterations
    @test argmax(ada_boost.alphas) == size(ada_boost.alphas, 1)

    @test_throws ArgumentError train_adaboost(X,y; iterations=0)
end

@testset "AdaBoost predict" begin
    test = [4.8, 3.0, 1.4, 0.3]
    true_label = 2

    preds = predict(ada_boost, test)

    @test size(preds, 1) == 1
end

