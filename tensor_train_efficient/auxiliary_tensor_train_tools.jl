using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress!, TruncBondThresh  



# Auxiliary function to create an identity tensor train of given length and physical dimensions
identity_tensor_train(N, qs) = [ones(1,1,qs...) for _ in 1:N] |> TensorTrain
identity_tensor_train(N,qs...) = identity_tensor_train(N,qs)
identity_tensor_train(A::AbstractTensorTrain) = identity_tensor_train(length(A), size(A[1])[3:end])



# Auxiliary function to estimate the norm of a tensor train
function estimate_norm_tt(B)
    return abs(B.z)*only(prod([maximum(abs(b[i,j,x]) for x in axes(b,3)) for i in axes(b,1), j in axes(b,2)] for b in B))
end


# Auxiliary functions to multiply and divide a TT by a constant
function divide_by_constant!(B, constant)
    B.z *= constant
    return B
end

function multiply_by_constant!(B, constant)
    B.z /= constant
    return B
end


# Auxiliary function to calculate the inverse of a tensor train using the Newton method
function inverse_tt(B, steps, bond)
    normalize_eachmatrix!(B)
    @show estimate_norm_tt(B)
    B0 = 1 / estimate_norm_tt(B)
    
    # Bn = B0 * (2I - B0 * B)
    temp = multiply_by_constant!(deepcopy(B), B0)
    two = multiply_by_constant!(identity_tensor_train(B), 2)
    Bn = multiply_by_constant!( two - temp, B0)
    
    #Bn = Bn - temp
    
    for t in 1:steps
        @show t

        # X_{n+1} = X_n * (2I - B * X_n)
        temp1 = B * Bn
        normalize_eachmatrix!(temp1)
        compress!(temp1; svd_trunc=TruncBond(bond))
        Bnn = two - temp1
        
        Bn = Bnn * Bn
    
        normalize_eachmatrix!(Bn)
        for b in Bn
            r = max(size(b,1), size(b,2))
        #    b ./= r
        #    Bn.z /= r
        end
        @show estimate_norm_tt(Bn)
        compress!(Bn; svd_trunc=TruncBond(bond))

    end
    
    return Bn
end

function inverse_tt_standard(B, bond)
    steps = log( estimate_norm_tt(B)) |> ceil |> Int
    return inverse_tt(B, steps, bond)
end


"""
* Kronecker multiplication of two tensors in Tensor Train format with matching physical dimensions.
"""
function Base.:*(A::T, B::T) where T<:AbstractTensorTrain
    C = map(zip(A.tensors, B.tensors)) do (a, b)  
        @assert size(a)[3:end] == size(b)[3:end]

        ar = reshape(a, size(a,1), size(a,2), prod(size(a)[3:end]))
        br = reshape(b, size(b,1), size(b,2), prod(size(b)[3:end]))
        
        @cast c[(ia,ib),(ja,jb),x] := ar[ia,ja,x] * br[ib,jb,x]

        reshape(c, size(c,1), size(c,2), size(a)[3:end]...)
    end

    T(C; z = A.z * B.z)
end


"""
    mult_sep(A, B): Multiplies two TensorTrains A and B by separating the physical dimensions.
"""
function mult_sep(A, B)
    d = map(zip(A.tensors,B.tensors)) do (a,b)
        @tullio c[m1,m2,n1,n2,x,y,x1,y1] := a[m1,n1,x,x1] * b[m2,n2,y,y1]
        @cast _[(m1,m2),(n1,n2),(x,y),(x1,y1)] := c[m1,m2,n1,n2,x,y,x1,y1]
    end
    return TensorTrain(d; z = A.z * B.z)   
end

