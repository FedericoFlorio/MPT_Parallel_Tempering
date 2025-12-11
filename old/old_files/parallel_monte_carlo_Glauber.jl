# ============================================
# ESTRUCTURA DE PARÁMETROS
# ============================================

"""
    GlauberParams

Parámetros del sistema de espines.

# Campos
- `beta::Float64`: Temperatura inversa (β = 1/kT)
- `j_vector::Vector{Float64}`: Acoplamientos J entre vecinos (longitud N-1)
- `h_vector::Vector{Float64}`: Campos magnéticos locales (longitud N)
- `p0::Float64`: Probabilidad de mantener el espín sin cambio
"""
struct GlauberParamsParallel
    beta_1::Float64
    beta_2::Float64
    j_vector::Vector{Float64}
    h_vector::Vector{Float64}
    p0::Float64
    
    function GlauberParamsParallel(beta_1, beta_2, j_vector, h_vector, p0=0.0)
        @assert length(j_vector) == length(h_vector) - 1 "j_vector debe tener longitud N-1"
        @assert 0 <= p0 <= 1 "p0 debe estar en [0,1]"
        new(beta_1, beta_2, j_vector, h_vector, p0)
    end
end

# ============================================
# TASA DE TRANSICIÓN (Dinámica de Glauber)
# ============================================

"""
    transition_rate(sigma_neighbors, sigma_new, site_index, params)

Calcula la probabilidad de transición P(σᵢᵗ⁺¹ = sigma_new | configuración actual).
"""
function transition_rate(sigma_neighbors, sigma_new, site_index, params)
    N = length(params.h_vector)
    
    if site_index == 1
        # Sitio 1: solo vecino derecho
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
        sigma_current = sigma_neighbors[1]
        
    elseif site_index == N
        # Sitio N: solo vecino izquierdo
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end]
        sigma_current = sigma_neighbors[2]
        
    else
        # Sitio intermedio: vecinos izquierdo y derecho
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
                params.j_vector[site_index] * sigma_neighbors[3] + 
                params.h_vector[site_index]
        sigma_current = sigma_neighbors[2]
    end
    
    # Dinámica de Glauber con probabilidad p0 de no cambiar
    glauber_prob = exp(params.beta * sigma_new * h_eff) / (2 * cosh(params.beta * h_eff))
    
    # Agregar componente de mantener el estado
    if sigma_new == sigma_current
        return (1 - params.p0) * glauber_prob + params.p0
    else
        return (1 - params.p0) * glauber_prob
    end
end

# ============================================
# INICIALIZACIÓN
# ============================================

"""
    initialize_spins(N, initial_probs)

Inicializa la cadena de espines con distribución producto independiente.

# Argumentos
- `N::Int`: Número de sitios
- `initial_probs::Vector{Float64}`: Probabilidades p_i de que σᵢ = +1 (longitud N)

# Retorna
- Vector de espines iniciales (±1)
"""
function initialize_spins(N::Int, initial_probs::Vector{Float64})
    @assert length(initial_probs) == N "initial_probs debe tener longitud N"
    @assert all(0 .<= initial_probs .<= 1) "Probabilidades deben estar en [0,1]"
    
    spins = zeros(Int, N)
    for i in 1:N
        spins[i] = rand() < initial_probs[i] ? 1 : -1
    end
    return spins
end

# ============================================
# PASO DE EVOLUCIÓN PARALELA
# ============================================

"""
    parallel_update!(spins_new, spins, params, rng)

Realiza un paso de actualización paralela (todos los espines simultáneamente).

# Argumentos
- `spins_new::Vector{Int}`: Vector para almacenar nueva configuración
- `spins::Vector{Int}`: Configuración actual
- `params::GlauberParams`: Parámetros del sistema
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `spins_new` con la nueva configuración
"""
function parallel_update!(spins_new, spins, params, rng)
    N = length(spins)
    
    for i in 1:N
        # Preparar vecinos según la posición
        if i == 1
            sigma_neighbors = [spins[1], spins[2]]
        elseif i == N
            sigma_neighbors = [spins[N-1], spins[N]]
        else
            sigma_neighbors = [spins[i-1], spins[i], spins[i+1]]
        end
        
        # Calcular probabilidades para σᵢ = +1 y σᵢ = -1
        p_up = transition_rate(sigma_neighbors, 1, i, params)
        p_down = transition_rate(sigma_neighbors, -1, i, params)
        
        # Normalizar (por seguridad numérica)
        p_total = p_up + p_down
        p_up /= p_total
        
        # Muestrear nuevo estado
        spins_new[i] = rand(rng) < p_up ? 1 : -1
    end
end

# ============================================
# SIMULACIÓN MONTE CARLO
# ============================================

"""
    SimulationResult

Resultado de la simulación Monte Carlo.

# Campos
- `trajectories::Array{Int,2}`: Trayectorias (N_samples × N × T_steps)
- `magnetizations::Matrix{Float64}`: Magnetización promedio por sitio (N_samples × N)
- `correlations::Matrix{Float64}`: Correlaciones espaciales promedio
- `params::GlauberParams`: Parámetros usados
"""
struct SimulationResultParallel
    trajectories_x::Array{Int,3}  # (N_samples, N, T_steps)
    trajectories_y::Array{Int,3}  # (N_samples, N, T_steps)
    magnetizations_x::Matrix{Float64}  # (N_samples, N)
    magnetizations_y::Matrix{Float64}  # (N_samples, N)
    params::GlauberParamsParallel
end

"""
    run_monte_carlo(N, params, initial_probs, T_steps; 
                    N_samples=1000, seed=123, save_trajectory=true)

Ejecuta simulación Monte Carlo de la dinámica de Glauber paralela.

# Argumentos
- `N::Int`: Número de espines
- `params::GlauberParams`: Parámetros del sistema
- `initial_probs::Vector{Float64}`: Distribución inicial P₀
- `T_steps::Int`: Número de pasos temporales

# Argumentos opcionales
- `N_samples::Int=1000`: Número de realizaciones independientes
- `seed::Int=123`: Semilla para reproducibilidad
- `save_trajectory::Bool=true`: Si guardar trayectorias completas

# Retorna
- `SimulationResult` con resultados de la simulación
"""
function run_parallel_monte_carlo(N::Int, params::GlauberParamsParallel, initial_probs::Vector{Float64}, 
                        T_steps::Int; N_samples::Int=1000, seed::Int=123, 
                        save_trajectory::Bool=true)
    
    rng = MersenneTwister(seed)
    
    params_1 = (N = N, beta = params.beta_1, j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    params_2 = (N = N, beta = params.beta_2, j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    
    # Almacenamiento
    if save_trajectory
        trajectories_x = zeros(Int, N_samples, N, T_steps + 1)
        trajectories_y = zeros(Int, N_samples, N, T_steps + 1)
    else
        trajectories_x = zeros(Int, 0, 0, 0)
        trajectories_y = zeros(Int, 0, 0, 0)
    end
    magnetizations_x = zeros(Float64, N_samples, N)
    magnetizations_y = zeros(Float64, N_samples, N)
    
    # Buffers para eficiencia
    spins_x = zeros(Int, N)
    spins_new_x = zeros(Int, N)
    
    spins_y = zeros(Int, N)
    spins_new_y = zeros(Int, N)

    # Simulación
    for sample in 1:N_samples
        # Inicializar
        spins_x .= initialize_spins(N, initial_probs)
        spins_y .= spins_x  # Misma configuración inicial para ambos sistemas
        
        if save_trajectory
            trajectories_x[sample, :, 1] .= spins_x
            trajectories_y[sample, :, 1] .= spins_y
        end
        
        # Evolución temporal
        for t in 1:T_steps
            parallel_update!(spins_new_x, spins_x, params_1, rng)
            parallel_update!(spins_new_y, spins_y, params_2, rng)
            spins_x, spins_new_x = spins_new_x, spins_x  # Swap eficiente
            spins_y, spins_new_y = spins_new_y, spins_y  # Swap eficiente
            
            if save_trajectory
                trajectories_x[sample, :, t+1] .= spins_x  
            end

            if save_trajectory
                trajectories_y[sample, :, t+1] .= spins_y 
            end
        end
        
        # Guardar magnetización final
        magnetizations_x[sample, :] .= spins_x
        magnetizations_y[sample, :] .= spins_y
    end
    
    return SimulationResultParallel(trajectories_x, trajectories_y, magnetizations_x, magnetizations_y, params)
end



# ============================================
# ANÁLISIS Y OBSERVABLES
# ============================================

"""
    compute_magnetization(result)

Calcula magnetización promedio por sitio.
"""
function compute_magnetization_parallel(result::SimulationResultParallel)
    return mean(result.magnetizations_x, dims=1)[1, :], mean(result.magnetizations_y, dims=1)[1, :]
end

"""
    compute_magnetization_error(result)

Calcula error estándar de la magnetización.
"""
function compute_magnetization_error_parallel(result::SimulationResultParallel)
    return std(result.magnetizations_x, dims=1)[1, :] / sqrt(size(result.magnetizations_x, 1)), std(result.magnetizations_y, dims=1)[1, :] / sqrt(size(result.magnetizations_y, 1))
end

"""
    compute_correlation(result, i, j)

Calcula ⟨σᵢ σⱼ⟩ promedio sobre realizaciones.
"""
function compute_correlation_parallel(result::SimulationResultParallel, i::Int, j::Int)
    return mean(result.magnetizations_x[:, i] .* result.magnetizations_x[:, j]), mean(result.magnetizations_y[:, i] .* result.magnetizations_y[:, j])
end

"""
    compute_marginal_magnetization_parallel(result)

Devuelve la magnetización marginal de cada espín en cada instante de tiempo, para ambos sistemas.
Retorna dos matrices: (mag_x, mag_y) de tamaño (N_x, T_x) y (N_y, T_y).
"""
function compute_marginal_magnetization_parallel(result)
    N_samples = size(result.trajectories_x, 1)
    N_x = size(result.trajectories_x, 2)
    T_x = size(result.trajectories_x, 3)
    N_y = size(result.trajectories_y, 2)
    T_y = size(result.trajectories_y, 3)
    mag_x = zeros(N_x, T_x)
    mag_y = zeros(N_y, T_y)
    for t in 1:T_x
        for i in 1:N_x
            mag_x[i, t] = mean(result.trajectories_x[sample, i, t] for sample in 1:N_samples)
        end
    end
    for t in 1:T_y
        for i in 1:N_y
            mag_y[i, t] = mean(result.trajectories_y[sample, i, t] for sample in 1:N_samples)
        end
    end
    return mag_x, mag_y
end
