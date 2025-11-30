@testset "best_split" begin
    # Perfect split found before final threshold
    feature = [1.0, 2.0, 10.0, 11.0] # thresholds: [1.5, 6.0, 10.5]
    labels = [0, 0, 1, 1]
    threshold, gini = best_split(feature, labels)
    @test threshold == 6
    @test gini == 0

    # Overlapping values
    feature2 = [1.0, 2.0, 3.0, 100.0] #thresholds: [1.5, 2.5, 51.5]
    labels2  = [0, 1, 1, 0]
    threshold2, gini2 = best_split(feature2, labels2)
    @test threshold2 == 1.5
    @test 0.32 <= gini2 <= 0.34
end