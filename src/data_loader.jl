function load_data(path::String) # use "src/data/Iris.csv" from within REPL
    df = CSV.read(path, DataFrame)

    p = shuffle(1:nrow(df))

    X = Matrix(df[p, 2:5])
    y = Vector(df.Species[p])

    return X, y
end