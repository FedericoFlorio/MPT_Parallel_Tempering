# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress, TruncBondThresh  




# ============================================================================
# TENSOR TRAIN DE EVOLUCIÓN DE LA DISTRIBUCIÓN DE PROBABILIDAD CON SWAP CON SALVA OPCIONAL
# ============================================================================
"""
    tensor_b_t_swap(A, P0, t, bond, s, save = true) Evoluciona la distribución de probabilidad inicial P0 a través de t 
    pasos de tiempo usando la matriz de transición A en formato TensorTrain, aplicando un swap con probabilidad s en cada paso.
    Devuelve una lista con el TensorTrain de la distribución en cada paso de tiempo si save es true.
# Argumentos
- `A`: TensorTrain que representa la matriz de transición
- `P0`: Vector de vectores con la distribución de probabilidad inicial en cada sitio
- `t`: Número de pasos de tiempo a evolucionar
- `bond`: Límite máximo para la compresión del TensorTrain
- `s`: Probabilidad de aplicar el swap en cada paso
- `save`: Booleano que indica si se guarda la distribución en cada paso
"""

function tensor_b_t_swap_acc_to_energy(A, P0, t, bond, swap_energy, save = true)
    B = TensorTrain([(@tullio _[1,1,x] := pi[x]) for pi in P0])
    lista_B_T = save ? [B] : nothing
    swap_idx = [1, 3, 2, 4]
    
    @showprogress for _ in 1:t
        # Evolución temporal
        B = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
            @tullio new_[m1,m2,n1,n2,σ_next] := A_i[m1,n1,σ,σ_next] * B_i[m2,n2,σ]
            @cast _[(m1,m2),(n1,n2),σ_next] := new_[m1,m2,n1,n2,σ_next]
        end |> TensorTrain
        
        # Crear B_swap sin deepcopy: solo reindexar cada tensor
        tensors_swap = [T[:, :, swap_idx] for T in B.tensors]
        B_swap = TensorTrain(tensors_swap; z = B.z)
        
        # Aplicar factores y sumar
        B_swap.tensors[1] *= s
        B.tensors[1] *= (1-s)
        B = B + B_swap
        
        compress!(B; svd_trunc=TruncBond(bond))
        normalize!(B)
        save && push!(lista_B_T, B)
    end
    
    return save ? lista_B_T : B
end

