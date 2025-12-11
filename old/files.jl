for dir in ["monte_carlo_efficient/", "tensor_train_efficient/"]
    for file in readdir(dir)
        if endswith(file, ".jl")
            include(joinpath(dir, file))
        end
    end
end