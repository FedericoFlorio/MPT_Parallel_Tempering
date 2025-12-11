# ============================================
# FUNCIÓN GENERAL DE MONTE CARLO
# ============================================

"""
    run_monte_carlo_general(
        params::MCParameters;
        initial_probs::Vector{Float64},            # Distribución inicial
        T_steps::Int,                              # Pasos temporales
        update_rule::Symbol = :parallel,           # :parallel o :sequential
        swap_criterion::Union{Nothing, Symbol} = nothing,  # nothing, :fixed_rate, :metropolis
        observables::Vector{Symbol} = [:magnetization],  # Lista de observables a calcular
        save_trajectory::Bool = false              # Guardar trayectorias completas
    )

Función general para simulaciones Monte Carlo en cadenas de Ising.

# Argumentos
- `params::MCParameters`: Parámetros del sistema (betas, j_vector, h_vector, etc.)
- `initial_probs::Vector{Float64}`: Probabilidades iniciales P(σᵢ = +1) (longitud N)
- `T_steps::Int`: Número de pasos temporales

# Keywords opcionales
- `update_rule::Symbol`: `:parallel` o `:sequential` (default: :parallel)
- `swap_criterion::Union{Nothing, Symbol}`: `nothing`, `:fixed_rate`, o `:metropolis` (default: nothing)
- `observables::Vector{Symbol}`: Observables a calcular (default: [:magnetization])
  Opciones: :magnetization, :energy, :nn_correlation, :overlap, :all_correlations
- `save_trajectory::Bool`: Guardar trayectorias completas (default: false)
"""

"""
# Ejemplos
julia
# Una cadena, dinámica paralela
params = MCParameters(beta=1.0, j_vector=ones(9), h_vector=zeros(10))
result = run_monte_carlo_general(
    params,
    initial_probs = fill(0.5, 10),
    T_steps = 100
)

# Dos cadenas con swap fijo
params = MCParameters(betas=[1.0, 2.0], j_vector=ones(9), h_vector=zeros(10), s=0.1)
result = run_monte_carlo_general(
    params,
    initial_probs = fill(0.5, 10),
    T_steps = 100,
    swap_criterion = :fixed_rate,
    observables = [:magnetization, :energy, :overlap]
)

# Dos cadenas con swap metropolis, dinámica secuencial
params = MCParameters(betas=[0.5, 1.5], j_vector=ones(9), h_vector=zeros(10))
result = run_monte_carlo_general(
    params,
    initial_probs = fill(0.5, 10),
    T_steps = 500,
    update_rule = :sequential,
    swap_criterion = :metropolis,
    observables = [:magnetization, :energy, :nn_correlation]
)

"""


function run_monte_carlo_general(
    params::MCParameters;
    initial_probs::Vector{Float64},
    T_steps::Int,
    update_rule::Symbol = :parallel,
    swap_criterion::Union{Nothing, Symbol} = nothing,
    observables::Vector{Symbol} = [:magnetization, :energy],
    save_trajectory::Bool = false,
    N_samples::Int = 1000
)
    # ============================================
    # VALIDACIÓN
    # ============================================
    
    validate(params)
    
    N = params.N
    n_chains_val = n_chains(params)
    
    @assert length(initial_probs) == N "initial_probs debe tener longitud N"
    @assert update_rule in [:parallel, :sequential, :metropolis] "update_rule debe ser :parallel, :sequential o :metropolis"
    
    # Validar swap
    if swap_criterion !== nothing
        @assert n_chains_val >= 2 "swap_criterion requiere al menos 2 cadenas"
        @assert swap_criterion in [:fixed_rate, :metropolis] "swap_criterion debe ser :fixed_rate o :metropolis"
    end
    
    # ============================================
    # INICIALIZACIÓN
    # ============================================
    
    rng = MersenneTwister(params.seed)
    
    # Inicializar cadenas (todas con la misma configuración inicial)
    chains = [initialize_spins(N, initial_probs, rng) for _ in 1:n_chains_val]
    

    # AGREGAR: Calcular energías iniciales
    chain_energies = [compute_total_energy(chains[c], params.j_vector, params.h_vector) 
                  for c in 1:n_chains_val]

    # Buffers solo necesarios para parallel update
    if update_rule == :parallel
        chains_new = [zeros(Int, N) for _ in 1:n_chains_val]
    else
        chains_new = nothing
    end
    
    # Inicializar acumuladores de observables
    obs_accumulators = initialize_observable_accumulators(observables, N, T_steps, 
                                                          N_samples, n_chains_val)
    
    # Opcional: guardar trayectorias (solo si se solicita)
    trajectories = save_trajectory ? [zeros(Int, N_samples, N, T_steps + 1) for _ in 1:n_chains_val] : nothing
    
    # ============================================
    # LOOP PRINCIPAL DE MUESTREO
    # ============================================
    
    for sample in 1:N_samples
        # Reinicializar todas las cadenas
        for c in 1:n_chains_val
            chains[c] .= initialize_spins(N, initial_probs, rng)
        end
        
        # Guardar estado inicial
        if save_trajectory
            for c in 1:n_chains_val
                trajectories[c][sample, :, 1] .= chains[c]
            end
        end
        
        # Acumular observables del estado inicial
        accumulate_observables!(obs_accumulators, chains, chain_energies, 1, sample, observables, 
                               params.j_vector, params.h_vector, params.betas)
        
        # ============================================
        # EVOLUCIÓN TEMPORAL
        # ============================================
        
        for t in 1:T_steps
            # Update de cada cadena según su criterio
            if update_rule == :parallel
                for c in 1:n_chains_val
                    parallel_update!(chains_new[c], chains[c], params.betas[c], 
                                    params.j_vector, params.h_vector, params.p0, rng)
                    chains[c], chains_new[c] = chains_new[c], chains[c]  # Swap eficiente

                    # AGREGAR: Recalcular energía (parallel cambia todo)
                    chain_energies[c] = compute_total_energy(chains[c], params.j_vector, params.h_vector)

                end
            elseif update_rule == :sequential
                for c in 1:n_chains_val
                    # sequential_update!(chains[c], params.betas[c], 
                    #                   params.j_vector, params.h_vector, params.p0, rng)
                    delta_E = sequential_update_with_energy!(chains[c], params.betas[c], 
                                                 params.j_vector, params.h_vector, params.p0, rng)
                    chain_energies[c] += delta_E

                end
            else  # :metropolis
                for c in 1:n_chains_val
                #     metropolis_update!(chains[c], params.betas[c], 
                #                       params.j_vector, params.h_vector, rng)
                    # MODIFICAR: retornar delta_E
                    accepted, delta_E = metropolis_update_with_energy!(chains[c], params.betas[c], 
                                                           params.j_vector, params.h_vector, rng)
                    if accepted
                        chain_energies[c] += delta_E
                    end
                end 
            end
            
            # Swap entre cadenas (si aplica)
            if swap_criterion !== nothing && n_chains_val >= 2
                if swap_criterion == :fixed_rate
                    # Fixed rate no necesita energías
                    apply_fixed_rate_swap!(chains, params.s, rng)
                    # AGREGAR: Después del swap, actualizar el orden de energías también
                    # (esto solo si apply_fixed_rate_swap! reordena chains)
                else  # :metropolis
                    # MODIFICAR: pasar energías precalculadas
                    apply_metropolis_swap_with_energies!(chains, chain_energies, params.betas, rng)
                end 
            end
            
            # Guardar estado
            if save_trajectory
                for c in 1:n_chains_val
                    trajectories[c][sample, :, t + 1] .= chains[c]
                end
            end
            
            # Acumular observables
            accumulate_observables!(obs_accumulators, chains, chain_energies, t, sample, observables,
                       params.j_vector, params.h_vector, params.betas)
        end
    end
    
    # ============================================
    # PROCESAR Y RETORNAR RESULTADOS
    # ============================================
    
    computed_observables = finalize_observables(obs_accumulators, N_samples, T_steps, n_chains_val)
    
    result = Dict{Symbol, Any}(
        :observables => computed_observables,
        :params => params
    )
    
    if save_trajectory
        result[:trajectories] = trajectories
    end
    
    return result
end


# ============================================
# FUNCIONES AUXILIARES
# ============================================

"""
    initialize_spins(N, initial_probs, rng)

Inicializa una cadena de espines según probabilidades dadas.
"""
function initialize_spins(N::Int, initial_probs::Vector{Float64}, rng)
    spins = zeros(Int, N)
    for i in 1:N
        spins[i] = rand(rng) < initial_probs[i] ? 1 : -1
    end
    return spins
end


# ============================================
# NOTA: Las siguientes funciones deben estar implementadas
# en archivos separados e incluidas antes de este archivo:
#
# De update_rules.jl:
#   - parallel_update!(spins_new, spins, beta, j_vector, h_vector, p0, rng)
#   - sequential_update!(spins, beta, j_vector, h_vector, p0, rng)
#   - metropolis_update!(spins, beta, j_vector, h_vector, rng)
#
# De swap_criteria.jl:
#   - apply_fixed_rate_swap!(chains, s, rng)
#   - apply_metropolis_swap!(chains, betas, j_vector, h_vector, rng)
#
# De observables.jl:
#   - initialize_observable_accumulators(observables, N, T_steps, N_samples, n_chains)
#   - accumulate_observables!(accumulators, chains, t, sample, observables, j_vector, h_vector, betas)
#   - finalize_observables(accumulators, N_samples, T_steps, n_chains)
#
# De parameters.jl:
#   - MCParameters (struct y constructor)
#   - validate(params)
#   - n_chains(params)
#   - N(params)
# ============================================