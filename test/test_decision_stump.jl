@testset "train_stump numeric test" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    stump = train_stump(X, y)

    @test stump.feature in 1:2
    @test stump.threshold isa Float64
    @test stump.left_label != stump.right_label
end


@testset "predict_stump test" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    stump = train_stump(X, y)
    preds = predict_stump(stump, X)

    @test preds == y
end
