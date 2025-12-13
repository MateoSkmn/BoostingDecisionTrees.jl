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

@testset "no split possible (1 sample) -> constant stump" begin
    X = reshape([5.0], 1, 1)   # 1x1 matrix
    y = [1]                    # single label

    stump = train_stump(X, y)

    @test stump.feature == 1
    @test stump.threshold == -Inf
    @test stump.left_label == 1
    @test stump.right_label == 1

    preds = predict_stump(stump, X)
    @test preds == [1]
end
