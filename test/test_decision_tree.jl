@testset "train_tree basic structure" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    tree = train_tree(X, y; max_depth=2)

    # Tree should be either a DecisionNode or LeafNode
    @test tree isa TreeNode
    
    # If it's a DecisionNode, check its structure
    if tree isa DecisionNode
        @test tree.feature in 1:2
        @test tree.threshold isa Float64
        @test tree.left isa TreeNode
        @test tree.right isa TreeNode
end


@testset "predict on training data" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    tree = train_tree(X, y; max_depth=5)
    preds = predict(tree, X)

    @test preds == y
    @test length(preds) == length(y)
    @test all(p in [0, 1] for p in preds)
end


@testset "predict single sample" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    tree = train_tree(X, y; max_depth=5)
    
    # Test single sample prediction
    pred_single = predict(tree, [1.0, 2.0])
    @test pred_single == 0
    
    pred_single2 = predict(tree, [10.0, 20.0])
    @test pred_single2 == 1
end


@testset "predict on matrix with single sample" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    tree = train_tree(X, y; max_depth=5)
    
    # Single sample as 1x2 matrix
    X_single = reshape([1.0, 2.0], 1, 2)
    preds = predict(tree, X_single)
    
    @test length(preds) == 1
    @test preds[1] == 0
end


@testset "no split possible (1 sample) -> leaf node" begin
    X = reshape([5.0, 5.0], 1, 2)   # 1x2 matrix
    y = [1]                         # single label

    tree = train_tree(X, y; max_depth=5)

    # Should return a leaf node with majority label
    @test tree isa LeafNode
    @test tree.label == 1

    # Predictions should always return the same label
    preds = predict(tree, X)
    @test preds == [1]
end


@testset "single class in dataset -> leaf node" begin
    X = [1  2;
         2  3;
         3  4]

    y = [0, 0, 0]  # All same class

    tree = train_tree(X, y; max_depth=5)

    @test tree isa LeafNode
    @test tree.label == 0

    preds = predict(tree, X)
    @test preds == [0, 0, 0]
end


@testset "max_depth constraint" begin
    X = [1  2;
         2  3;
         3  4;
         4  5;
         10 20;
         11 21;
         12 22;
         13 23]

    y = [0, 0, 0, 0, 1, 1, 1, 1]

    # Train with max_depth=1 (stumps only)
    tree_depth1 = train_tree(X, y; max_depth=1)
    
    # With max_depth=1, we can have at most 2 levels: root + leaves
    # So left and right should be LeafNodes
    @test tree_depth1 isa DecisionNode
    @test tree_depth1.left isa LeafNode
    @test tree_depth1.right isa LeafNode

    # Train with max_depth=3
    tree_depth3 = train_tree(X, y; max_depth=3)
    @test tree_depth3 isa TreeNode
end


@testset "max_depth=0 returns leaf with majority label" begin
    X = [1  2;
         2  3;
         3  4;
         10 20]

    y = [0, 0, 0, 1]

    tree = train_tree(X, y; max_depth=0)

    @test tree isa LeafNode
    @test tree.label == 0  # Majority label
end


@testset "tree with numeric and string labels" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = ["a", "a", "a", "b", "b", "b"]

    tree = train_tree(X, y; max_depth=5)
    preds = predict(tree, X)

    @test preds == y
    @test all(p in ["a", "b"] for p in preds)
end


@testset "overlapping features" begin
    X = [1.0  1.0;
         1.5  1.5;
         2.0  2.0;
         100.0  100.0;
         100.5  100.5;
         101.0  101.0]

    y = [0, 0, 0, 1, 1, 1]

    tree = train_tree(X, y; max_depth=5)
    preds = predict(tree, X)

    @test preds == y
end


@testset "leaf node prediction consistency" begin
    # Create a simple leaf node and test it
    leaf = LeafNode("test_label")
    
    x_sample = [1.0, 2.0, 3.0]
    pred = predict(leaf, x_sample)
    
    @test pred == "test_label"
end


@testset "decision node routing" begin
    # Create a simple manually constructed tree:
    #       feature=1, threshold=2.5
    #      /                        \
    #   "left"                    "right"
    
    left_leaf = LeafNode("left")
    right_leaf = LeafNode("right")
    tree = DecisionNode(1, 2.5, left_leaf, right_leaf)
    
    # Test routing to left
    pred_left = predict(tree, [1.0, 10.0])  # feature 1 is 1.0, <= 2.5
    @test pred_left == "left"
    
    # Test routing to right
    pred_right = predict(tree, [3.0, 10.0])  # feature 1 is 3.0, > 2.5
    @test pred_right == "right"
    
    # Test boundary
    pred_boundary = predict(tree, [2.5, 10.0])  # feature 1 is 2.5, <= 2.5
    @test pred_boundary == "left"
end


@testset "predict output type" begin
    X = [1  2;
         2  3;
         10 20;
         11 21]

    y = [0, 0, 1, 1]

    tree = train_tree(X, y; max_depth=5)
    preds = predict(tree, X)

    # Should return Vector{Any}
    @test isa(preds, Vector)
    @test length(preds) == size(X, 1)
end


@testset "no valid split found (all features return nothing)" begin
    # Create a dataset where best_split returns nothing for all features
    # This happens when we recursively split and get data where all values are identical
    # Use data that creates a split, then one side becomes constant
    X = [1.0  1.0;
         1.0  1.0;
         10.0  10.0;
         10.0  10.0]

    y = [0, 0, 1, 1]

    tree = train_tree(X, y; max_depth=5)

    # Should successfully split initially since we have variation
    @test tree isa DecisionNode
    
    # Both subtrees should be leaf nodes since after the split,
    # all samples in each partition have the same label
    @test tree.left isa LeafNode
    @test tree.right isa LeafNode

    preds = predict(tree, X)
    @test preds == y
end


@testset "Gini no valid split" begin
    X = [1.0 1.0]

    y = [1]

    tree = train_tree(X, y; criterion=:gini)

    @test tree isa LeafNode
    @test tree.label == 1  # majority label
end

@testset "mixed case: some features split, some don't" begin
    # One feature has variation, another doesn't
    X = [5.0  1.0;
         5.0  2.0;
         5.0  3.0;
         5.0  100.0;
         5.0  101.0;
         5.0  102.0]

    y = [0, 0, 0, 1, 1, 1]

    tree = train_tree(X, y; max_depth=5)

    # Should successfully find a split on feature 2 (column 2)
    # even though feature 1 has no variation
    @test tree isa DecisionNode
    @test tree.feature == 2

    preds = predict(tree, X)
    @test preds == y
end

@testset "use information gain criterion" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    tree = train_tree(X, y; max_depth=5,criterion=:information_gain)
    preds = predict(tree, X)

    @test preds == y
    @test length(preds) == length(y)
    @test all(p in [0, 1] for p in preds)
end
    
@testset "information gain score < 0" begin
    # This happens when we recursively split and get data where all values are identical
    # Use data that creates a split, then one side becomes constant
    X = [1.0  1.0;
         1.0  1.0;
         1.0  1.0;
         1.0  1.0]

    y = [0, 0, 1, 1]
 
    tree = train_tree(X, y; max_depth=5, criterion=:information_gain)

    # Should successfully split initially since we have variation
    @test tree isa LeafNode
end

@testset "unknown criterion" begin
    X = [1  2;
         2  3;
         3  4;
         10 20;
         11 21;
         12 22]

    y = [0, 0, 0, 1, 1, 1]

    @test_throws ArgumentError train_tree(X, y; max_depth=5,criterion=:unknown)
end

end