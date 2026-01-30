@testset "Iris Data Preparation Tests" begin
    X,y = load_data_iris("../src/data/Iris.csv")
    # X is 150x4 - y has 150 elements
    @test size(X) == (150, 4)
    @test length(y) == 150
end