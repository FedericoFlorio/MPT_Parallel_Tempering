# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress!, TruncBondThresh  



function tt_swap(params, n, bond)
    tt = boltzman_tt(params)
    for i in 2:n
        tt += boltzman_n_tt(params, i) 
        compress!(tt; svd_trunc=TruncBond(bond))
    end
    normalize_eachmatrix!(tt)
    tt_inverse = inverse_tt_standard(tt + identity_tensor_train(tt), bond)
    return tt * tt_inverse
end

