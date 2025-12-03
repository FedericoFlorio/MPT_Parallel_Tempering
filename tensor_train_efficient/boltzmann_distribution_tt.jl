# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress!, TruncBondThresh  




##################################################################################
##################################################################################

# BOLTZMAN DISTRIBUTION TT

##################################################################################
##################################################################################


function boltzman_tt(params, Q::Int = 2, σ = x -> 2x - 3)
    # Inferir N del tamaño de h_vector
    N = length(params.h_vector)
    
    function delta_vector(spin_value)
        return [spin_value == σ(q) ? 1.0 : 0.0 for q in 1:Q]'
    end

    tensors = Array{Float64, 3}[]  # Vector de arrays 3D
    
    h_vector = params.h_vector
    j_vector = params.j_vector
    beta = hasproperty(params, :beta_1) ? params.beta_1 : params.beta

    # ============================================
    # Tensor inicial A1
    # ============================================
    A1 = zeros(1, Q, Q)
    
    for sigma in 1:Q          # σ₁ᵗ
        external_force = exp(beta * h_vector[1] * σ(sigma))
        # Vector fila f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ) de dimensión 1×Q
        # ωⱼ representa los posibles valores de σ₂ᵗ
        f_vector = zeros(1, Q)
        for j in 1:Q
            omega_j = σ(j)  # Valor de σ₂ᵗ
            f_vector[1, j] = exp(beta * j_vector[1] * σ(sigma) * omega_j) * external_force
        end
        
        A1[1, :, sigma] = f_vector 
    end
    
    push!(tensors, A1)
    
    # ============================================
    # Tensores intermedios A2, ..., A_{N-1}
    # ============================================
    for i in 2:N-1
        Ai = zeros(Q, Q, Q)
        
        for sigma in 1:Q         # σᵢᵗ
            external_force = exp(beta * h_vector[i] * σ(sigma))
            left_factor = delta_vector( σ( sigma))
            right_factor = zeros(1, Q)

            for j in 1:Q
                omega_j = σ(j)  # Valor de σᵢ₊₁ᵗ
                right_factor[1, j] = exp(beta * j_vector[i] * σ(sigma) * omega_j) * external_force
            end

            Ai[:, :, sigma] = left_factor' * right_factor
        end
        
        push!(tensors, Ai)
    end
    
    # ============================================
    # Tensor final AN
    # ============================================
    AN = zeros(Q, 1, Q)
    
    for sigma in 1:Q    
        external_force = exp(beta * h_vector[N] * σ(sigma))
        AN[:, 1, sigma] = delta_vector(σ( sigma))' * external_force
    end
    
    push!(tensors, AN)
    
    # Crear y retornar el TensorTrain
    return TensorTrain(tensors)
end



function boltzman_n_tt(params, n) 
    params.beta = n * params.beta
    return boltzman_tt(params)
end


