# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress, TruncBondThresh  




# ============================================================================
# TENSOR TRAIN DE EVOLUCIÓN DE LA DISTRIBUCIÓN DE PROBABILIDAD
# ============================================================================

"""
    tensor_b_t(A, P0, t, bond) Evoluciona la distribución de probabilidad inicial P0 a través de t 
    pasos de tiempo usando la matriz de transición A en formato TensorTrain.
# Argumentos
- `A`: TensorTrain que representa la matriz de transición
- `P0`: Vector de vectores con la distribución de probabilidad inicial en cada sitio
- `t`: Número de pasos de tiempo a evolucionar
- `bond`: Límite para la compresión del TensorTrain
"""
function tensor_b_t(A, P0, t, bond)
    N = length(A.tensors)               # Define N como la longitud de A.tensors

    # Construye el TensorTrain inicial para la distribución P0
    # Para cada sitio, crea un tensor de tamaño (1,1,Q) con las probabilidades iniciales.
    B = TensorTrain([(@tullio _[1,1,x] := pi[x]) for pi in P0])        

    # Itera sobre los pasos de tiempo, mostrando una barra de progreso.
    @showprogress for step in 1:t   

        # Para cada sitio, toma el tensor de transición Ai y el tensor de probabilidad Bi
        B = map(zip(A.tensors,B.tensors)) do (A_i, B_i)     
            
            # Realiza la suma sobre σ_t (el estado anterior), multiplicando el tensor de transición 
            # por la distribución. El resultado es un nuevo tensor para el siguiente tiempo.
            @tullio new_tensor_[m1,m2,n1,n2,sigma_t_plus] := A_i[m1,n1,sigma_t,sigma_t_plus] * B_i[m2,n2,sigma_t]

            # Reordena las dimensiones para que los bonds estén agrupados correctamente.
            @cast _[(m1,m2),(n1,n2),sigma_t_plus] := new_tensor_[m1,m2,n1,n2,sigma_t_plus]

        # Crea el nuevo TensorTrain con los tensores actualizados.
        end |> TensorTrain
        compress!(B; svd_trunc=TruncBond(bond)) 
        normalize!(B)
    end
    
    return B
end

# ============================================================================

# ============================================================================================
# TENSOR TRAIN DE EVOLUCIÓN DE LA DISTRIBUCIÓN DE PROBABILIDAD, COS SALVA PARA CADA T
# ============================================================================================

"""
    tensor_b_t(A, P0, t, bond) Evoluciona la distribución de probabilidad inicial P0 a través de t 
    pasos de tiempo usando la matriz de transición A en formato TensorTrain. Devuelve una lista con el 
    TensorTrain de la distribución en cada paso de tiempo.
# Argumentos
- `A`: TensorTrain que representa la matriz de transición
- `P0`: Vector de vectores con la distribución de probabilidad inicial en cada sitio
- `t`: Número de pasos de tiempo a evolucionar
- `bond`: Límite máximo para la compresión del TensorTrain
"""
function tensor_b_t_over_time(A, P0, t, bond)
    N = length(A.tensors)               # Define N como la longitud de A.tensors

    lista_B_T =[]

    # Construye el TensorTrain inicial para la distribución P0
    # Para cada sitio, crea un tensor de tamaño (1,1,Q) con las probabilidades iniciales.
    B = TensorTrain([(@tullio _[1,1,x] := pi[x]) for pi in P0])        
    push!(lista_B_T, B)
    # Itera sobre los pasos de tiempo, mostrando una barra de progreso.
    @showprogress for step in 1:t   

        # Para cada sitio, toma el tensor de transición Ai y el tensor de probabilidad Bi
        B = map(zip(A.tensors,B.tensors)) do (A_i, B_i)     
            
            # Realiza la suma sobre σ_t (el estado anterior), multiplicando el tensor de transición 
            # por la distribución. El resultado es un nuevo tensor para el siguiente tiempo.
            @tullio new_tensor_[m1,m2,n1,n2,sigma_t_plus] := A_i[m1,n1,sigma_t,sigma_t_plus] * B_i[m2,n2,sigma_t]

            # Reordena las dimensiones para que los bonds estén agrupados correctamente.
            @cast _[(m1,m2),(n1,n2),sigma_t_plus] := new_tensor_[m1,m2,n1,n2,sigma_t_plus]

        # Crea el nuevo TensorTrain con los tensores actualizados.
        end |> TensorTrain
        compress!(B; svd_trunc=TruncBond(bond)) 
        normalize!(B)
        push!(lista_B_T, B)
    end
    
    return lista_B_T
end



# ============================================
# FUNCIONES AUXILIARES
# ============================================

"""
σ(x) Mapea índice de espín a valor físico: 1 → -1, 2 → +1
"""
σ(x) = 2x - 3


# ============================================
# SELECCIÓN DE PARÁMETROS ALEATORIOS
# ============================================

"""
    random_params(N): Genera parámetros aleatorios para el modelo de Ising.
"""
function random_params(N)
    a, b = -1.0, 1.0
    params = (
        N = N, 
        beta = rand(),                              # Inversa de la temperatura (β = 1/kT)
        j_vector = a .+ (b - a) .* rand(N-1) ,        # Acoplamientos J_{i,i+1} (N-1 elementos)
        h_vector = a .+ (b - a) .* rand(N) ,    # Campos externos h_i (N elementos)
        p0 = rand(),                                 # Probabilidad de mantener configuración,
    )
    return params
end

"""
    random_params(N): Genera parámetros aleatorios para el modelo de Ising paralelo.
"""
function parallel_random_params(N)
    a, b = -1.0, 1.0
    params = (
        N = N, 
        beta_1 = rand(),                              # Inversa de la temperatura (β = 1/kT)
        beta_2 = rand(),                              # Inversa de la temperatura (β = 1/kT)
        j_vector = a .+ (b - a) .* rand(N-1) ,        # Acoplamientos J_{i,i+1} (N-1 elementos)
        h_vector = a .+ (b - a) .* rand(N) ,          # Campos externos h_i (N elementos)
        p0 = rand()                                   # Probabilidad de mantener configuración,
    )
    return params
end

"""
    random_P0(N, Q): Genera una distribución de probabilidad inicial aleatoria normalizada para N sitios 
    y Q estados por espín.
"""
function random_P0(N, Q = 2)
    P0 = [rand(Q) for _ in 1:N]
    for i in 1:N
        P0[i] /= sum(P0[i])  # Normaliza cada vector de probabilidad
    end
    return P0
end


"""
    parallel_random_P0_fixed(N): Genera una distribución de probabilidad inicial fija para N sitios 
    en el modelo paralelo.
"""
function parallel_random_P0_fixed(N)
    P0 = [Float64[rand(), 0.0, 0.0, rand()] for _ in 1:N]
    for i in 1:N
        P0[i] ./= sum(P0[i])  # Normaliza cada vector de probabilidad
    end
    return P0
end
    


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

function tensor_b_t_swap(A, P0, t, bond, s, save = true)
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







#swap(i) = (i == 1 ? 1 : i == 2 ? 3 : i == 3 ? 2 : i == 4 ? 4 : error("swap solo definido para i=1..4"))


# function doble_tensor_b_t_tplus_swap(A, B, bond, s)
#     swap_A_idx = [1,2,3,4, 9,10,11,12, 5,6,7,8, 13,14,15,16]
    
#     # Crear B_swap sin deepcopy: solo reindexar cada tensor
#     tensors_A_swap = [T[:, :, swap_A_idx] for T in A.tensors]
#     A_swap = TensorTrain(tensors_A_swap; z = A.z) 
    
#     # Evolución temporal
#     B_ = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
#         @tullio new_[m1,m2,n1,n2,σ, σ_next] := A_i[m1,n1,σ,σ_next] * B_i[m2,n2,σ]
#         @cast _[(m1,m2),(n1,n2),(σ,σ_next)] := new_[m1,m2,n1,n2,σ,σ_next]
#     end |> TensorTrain

#     B_swap = map(zip(A_swap.tensors, B.tensors)) do (A_i, B_i)
#         @tullio new_[m1,m2,n1,n2,σ, σ_next] := A_i[m1,n1,σ,σ_next] * B_i[m2,n2,σ]
#         @cast _[(m1,m2),(n1,n2),(σ, σ_next)] := new_[m1,m2,n1,n2,σ, σ_next]
#     end |> TensorTrain
    
#     # Aplicar factores y sumar
#     B_swap.tensors[1] *= s
#     B_.tensors[1] *= (1-s)
#     B = B_ + B_swap
    
#     compress!(B; svd_trunc=TruncBond(bond))
#     normalize!(B)
    
#     return B
# end
