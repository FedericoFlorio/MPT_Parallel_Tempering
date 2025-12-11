# ============================================
# SISTEMA DE OBSERVABLES
# ============================================

# Este archivo implementa un sistema eficiente para calcular observables
# on-the-fly durante las simulaciones Monte Carlo, sin necesidad de
# guardar trayectorias completas.

# ============================================
# INICIALIZACIÓN DE ACUMULADORES
# ============================================

"""
    initialize_observable_accumulators(observables, N, T_steps, N_samples, n_chains)

Inicializa estructuras para acumular observables durante la simulación.

# Argumentos
- `observables::Vector{Symbol}`: Lista de observables a calcular
- `N::Int`: Número de espines por cadena
- `T_steps::Int`: Número de pasos temporales
- `N_samples::Int`: Número de muestras
- `n_chains::Int`: Número de cadenas

# Retorna
- `Dict{Symbol, Any}`: Diccionario con acumuladores para cada observable

# Observables soportados
- `:magnetization`: Magnetización por sitio ⟨σᵢ(t)⟩
- `:energy`: Energía total ⟨E(t)⟩
- `:nn_correlation`: Correlación entre vecinos cercanos ⟨σᵢ σᵢ₊₁⟩
- `:overlap`: Overlap entre réplicas (solo si n_chains ≥ 2)
- `:all_correlations`: Matriz completa de correlaciones ⟨σᵢ σⱼ⟩
"""
function initialize_observable_accumulators(observables, N, T_steps, N_samples, n_chains)
    accumulators = Dict{Symbol, Any}()
    
    for obs in observables
        if obs == :magnetization
            # Acumula ⟨σᵢ(t)⟩ para cada cadena
            # Dimensiones: (n_chains, N, T_steps+1)
            accumulators[:magnetization] = zeros(Float64, n_chains, N, T_steps + 1)
            
        elseif obs == :energy
            # Acumula ⟨E(t)⟩ para cada cadena
            # Dimensiones: (n_chains, T_steps+1)
            accumulators[:energy] = zeros(Float64, n_chains, T_steps + 1)
            accumulators[:energy_sq] = zeros(Float64, n_chains, T_steps + 1)  # Para varianza
            
        elseif obs == :nn_correlation
            # Acumula ⟨σᵢ σᵢ₊₁⟩ promediado sobre enlaces
            # Dimensiones: (n_chains, T_steps+1)
            accumulators[:nn_correlation] = zeros(Float64, n_chains, T_steps + 1)
            
        elseif obs == :overlap
            if n_chains >= 2
                # Acumula overlap Q(t) = (1/N) ∑ᵢ ⟨σᵢ¹(t) σᵢ²(t)⟩
                # Para todas las combinaciones de pares de cadenas
                # Dimensiones: (n_pairs, T_steps+1) donde n_pairs = n_chains*(n_chains-1)/2
                n_pairs = div(n_chains * (n_chains - 1), 2)
                accumulators[:overlap] = zeros(Float64, n_pairs, T_steps + 1)
                accumulators[:overlap_pairs] = [(i,j) for i in 1:n_chains for j in (i+1):n_chains]
            end
            
        elseif obs == :all_correlations
            # Acumula ⟨σᵢ σⱼ⟩ para todos los pares
            # ADVERTENCIA: Esto puede usar mucha memoria para N grande
            # Dimensiones: (n_chains, N, N, T_steps+1)
            accumulators[:all_correlations] = zeros(Float64, n_chains, N, N, T_steps + 1)
            
        else
            @warn "Observable $obs no reconocido, será ignorado"
        end
    end
    
    return accumulators
end

# ============================================
# ACUMULACIÓN DE OBSERVABLES
# ============================================

"""
    accumulate_observables!(accumulators, chains, t, sample, observables, 
                           j_vector, h_vector, betas)

Acumula valores de observables para el estado actual de las cadenas.

# Argumentos
- `accumulators::Dict`: Diccionario de acumuladores (modificado in-place)
- `chains::Vector{Vector{Int}}`: Estado actual de todas las cadenas
- `t::Int`: Índice de tiempo (1 a T_steps+1)
- `sample::Int`: Índice de muestra actual (1 a N_samples)
- `observables::Vector{Symbol}`: Lista de observables a acumular
- `j_vector::Vector{Float64}`: Acoplamientos
- `h_vector::Vector{Float64}`: Campos magnéticos
- `betas::Vector{Float64}`: Temperaturas inversas

# Modifica
- Los acumuladores en `accumulators` sumando las contribuciones del estado actual
"""
function accumulate_observables!(accumulators, chains, energies, t, sample, observables,
                                j_vector, h_vector, betas)
    N = length(chains[1])
    n_chains = length(chains)
    
    for obs in observables
        if obs == :magnetization
            # Acumular magnetización por sitio
            for c in 1:n_chains
                for i in 1:N
                    accumulators[:magnetization][c, i, t] += chains[c][i]
                end
            end
            
        elseif obs == :energy
            # Acumular energía total
            for c in 1:n_chains
                E = compute_total_energy(chains[c], j_vector, h_vector)
                # E = energies[c] 
                accumulators[:energy][c, t] += E
                accumulators[:energy_sq][c, t] += E^2
            end
            
        elseif obs == :nn_correlation
            # Acumular correlación entre vecinos cercanos
            for c in 1:n_chains
                nn_corr = 0.0
                for i in 1:(N-1)
                    nn_corr += chains[c][i] * chains[c][i+1]
                end
                accumulators[:nn_correlation][c, t] += nn_corr / (N - 1)
            end
            
        elseif obs == :overlap
            if n_chains >= 2
                # Acumular overlap entre pares de cadenas
                pair_idx = 1
                for i in 1:n_chains
                    for j in (i+1):n_chains
                        overlap_ij = 0.0
                        for site in 1:N
                            overlap_ij += chains[i][site] * chains[j][site]
                        end
                        accumulators[:overlap][pair_idx, t] += overlap_ij / N
                        pair_idx += 1
                    end
                end
            end
            
        elseif obs == :all_correlations
            # Acumular todas las correlaciones por pares
            for c in 1:n_chains
                for i in 1:N
                    for j in 1:N
                        accumulators[:all_correlations][c, i, j, t] += 
                            chains[c][i] * chains[c][j]
                    end
                end
            end
        end
    end
end

# ============================================
# FINALIZACIÓN Y CÁLCULO DE OBSERVABLES
# ============================================

"""
    finalize_observables(accumulators, N_samples, T_steps, n_chains)

Procesa los acumuladores para obtener los observables finales.

# Argumentos
- `accumulators::Dict`: Diccionario con valores acumulados
- `N_samples::Int`: Número de muestras (para normalización)
- `T_steps::Int`: Número de pasos temporales
- `n_chains::Int`: Número de cadenas

# Retorna
- `Dict{Symbol, Any}`: Diccionario con observables finalizados

# Procesamiento realizado
- Divide por N_samples para obtener promedios
- Calcula errores estándar donde es relevante
- Calcula varianzas y susceptibilidades
"""
function finalize_observables(accumulators, N_samples, T_steps, n_chains)
    results = Dict{Symbol, Any}()
    
    for (obs_name, acc_data) in accumulators
        if obs_name == :magnetization
            # Normalizar por N_samples
            mag = acc_data ./ N_samples
            results[:magnetization] = mag
            
            # Para cada cadena, retornar matriz (N, T_steps+1)
            if n_chains == 1
                results[:magnetization] = mag[1, :, :]
            end
            
        elseif obs_name == :energy
            # Energía promedio
            energy_mean = accumulators[:energy] ./ N_samples
            energy_sq_mean = accumulators[:energy_sq] ./ N_samples
            
            # Varianza: ⟨E²⟩ - ⟨E⟩²
            energy_var = energy_sq_mean .- energy_mean.^2
            
            # Error estándar: σ/√N_samples
            energy_std = sqrt.(max.(energy_var, 0.0))  # max para evitar negativos por error numérico
            energy_error = energy_std ./ sqrt(N_samples)
            
            results[:energy] = energy_mean
            results[:energy_error] = energy_error
            results[:energy_variance] = energy_var
            
            # Para una sola cadena, simplificar
            if n_chains == 1
                results[:energy] = energy_mean[1, :]
                results[:energy_error] = energy_error[1, :]
                results[:energy_variance] = energy_var[1, :]
            end
            
        elseif obs_name == :nn_correlation
            # Normalizar correlación de vecinos cercanos
            nn_corr = acc_data ./ N_samples
            results[:nn_correlation] = nn_corr
            
            if n_chains == 1
                results[:nn_correlation] = nn_corr[1, :]
            end
            
        elseif obs_name == :overlap
            # Normalizar overlap
            overlap = acc_data ./ N_samples
            results[:overlap] = overlap
            
            # Incluir información sobre qué pares corresponden a cada fila
            if haskey(accumulators, :overlap_pairs)
                results[:overlap_pairs] = accumulators[:overlap_pairs]
            end
            
        elseif obs_name == :all_correlations
            # Normalizar matriz completa de correlaciones
            corr = acc_data ./ N_samples
            results[:all_correlations] = corr
            
            if n_chains == 1
                results[:all_correlations] = corr[1, :, :, :]
            end
        end
    end
    
    return results
end

# ============================================
# FUNCIONES AUXILIARES PARA OBSERVABLES
# ============================================

"""
    compute_total_energy(spins, j_vector, h_vector)

Calcula la energía total de una configuración.
(Duplicado de swap_criteria.jl para independencia del módulo)
"""
function compute_total_energy(spins, j_vector, h_vector)
    N = length(spins)
    energy = 0.0
    
    # Término de interacción
    for i in 1:(N-1)
        energy -= j_vector[i] * spins[i] * spins[i+1]
    end
    
    # Término de campo magnético
    for i in 1:N
        energy -= h_vector[i] * spins[i]
    end
    
    return energy
end



































# ============================================
# ANÁLISIS POST-PROCESAMIENTO
# ============================================

"""
    compute_susceptibility(observables, betas)

Calcula la susceptibilidad magnética χ = β⟨(M - ⟨M⟩)²⟩ a partir de los observables.

# Nota
Requiere que se haya calculado :magnetization durante la simulación.
Si no se guardaron trayectorias, esto solo puede calcularse aproximadamente.
"""


# function compute_susceptibility(magnetization, betas, N)
#     if ndims(magnetization) == 2
#         # Una sola cadena: magnetization es (N, T_steps+1)
#         M_total = sum(magnetization, dims=1)[1, :]  # Magnetización total por tiempo
#         chi = betas[1] * var(M_total)
#         return chi
#     else
#         # Múltiples cadenas: magnetization es (n_chains, N, T_steps+1)
#         n_chains = size(magnetization, 1)
#         chi = zeros(n_chains)
        
#         for c in 1:n_chains
#             M_total = sum(magnetization[c, :, :], dims=1)[1, :]
#             chi[c] = betas[c] * var(M_total)
#         end
        
#         return chi
#     end
# end



"""
    compute_specific_heat(energy_variance, betas)

Calcula la capacidad calorífica C = β²⟨(E - ⟨E⟩)²⟩.

# Nota
Requiere que se haya calculado :energy durante la simulación.
energy_variance ya está calculada en finalize_observables.
"""


# function compute_specific_heat(energy_variance, betas)
#     if energy_variance isa Vector
#         # Una sola cadena
#         return betas[1]^2 .* energy_variance
#     else
#         # Múltiples cadenas
#         n_chains = size(energy_variance, 1)
#         C = zeros(size(energy_variance))
        
#         for c in 1:n_chains
#             C[c, :] = betas[c]^2 .* energy_variance[c, :]
#         end
        
#         return C
#     end
# end

"""
    compute_autocorrelation(magnetization_timeseries; max_lag=nothing)

Calcula la función de autocorrelación temporal de la magnetización.

# Argumentos
- `magnetization_timeseries::Vector{Float64}`: Serie temporal de magnetización total
- `max_lag::Union{Nothing, Int}`: Máximo lag a calcular (default: length÷4)

# Retorna
- `Vector{Float64}`: Función de autocorrelación normalizada C(τ)/C(0)
"""
# function compute_autocorrelation(magnetization_timeseries; max_lag=nothing)
#     T = length(magnetization_timeseries)
    
#     if max_lag === nothing
#         max_lag = div(T, 4)
#     end
    
#     mean_mag = mean(magnetization_timeseries)
    
#     # Calcular autocorrelación
#     autocorr = zeros(max_lag + 1)
    
#     for lag in 0:max_lag
#         sum_prod = 0.0
#         count = 0
#         for t in 1:(T - lag)
#             sum_prod += (magnetization_timeseries[t] - mean_mag) * 
#                        (magnetization_timeseries[t + lag] - mean_mag)
#             count += 1
#         end
#         autocorr[lag + 1] = sum_prod / count
#     end
    
#     # Normalizar por C(0)
#     autocorr ./= autocorr[1]
    
#     return autocorr
# end

"""
    estimate_autocorrelation_time(autocorr; threshold=exp(-1))

Estima el tiempo de autocorrelación τ a partir de la función de autocorrelación.

# Argumentos
- `autocorr::Vector{Float64}`: Función de autocorrelación
- `threshold::Float64`: Umbral para definir τ (default: 1/e)

# Retorna
- `Int`: Tiempo de autocorrelación estimado
"""
# function estimate_autocorrelation_time(autocorr; threshold=exp(-1))
#     tau_idx = findfirst(x -> x <= threshold, autocorr)
    
#     if tau_idx === nothing
#         return length(autocorr)  # No decayó suficiente
#     else
#         return tau_idx - 1
#     end
# end

# ============================================
# FUNCIONES DE CONVENIENCIA
# ============================================

"""
    extract_observable_at_time(observables, obs_name, t; chain=nothing)

Extrae el valor de un observable en un tiempo específico.

# Argumentos
- `observables::Dict`: Diccionario de observables finalizados
- `obs_name::Symbol`: Nombre del observable
- `t::Int`: Índice de tiempo
- `chain::Union{Nothing, Int}`: Cadena específica (si hay múltiples)

# Retorna
- Valor del observable en el tiempo t
"""
# function extract_observable_at_time(observables, obs_name, t; chain=nothing)
#     if !haskey(observables, obs_name)
#         error("Observable $obs_name no fue calculado")
#     end
    
#     data = observables[obs_name]
    
#     # Manejar diferentes dimensionalidades
#     if obs_name == :magnetization
#         if ndims(data) == 2
#             # Una cadena: (N, T)
#             return data[:, t]
#         else
#             # Múltiples cadenas: (n_chains, N, T)
#             if chain === nothing
#                 return data[:, :, t]
#             else
#                 return data[chain, :, t]
#             end
#         end
        
#     elseif obs_name in [:energy, :nn_correlation]
#         if data isa Vector
#             # Una cadena
#             return data[t]
#         else
#             # Múltiples cadenas
#             if chain === nothing
#                 return data[:, t]
#             else
#                 return data[chain, t]
#             end
#         end
        
#     else
#         # Para otros observables, retornar lo que sea apropiado
#         return data
#     end
# end

# ============================================
# INFORMACIÓN
# ============================================

"""
Sistema de observables eficiente:

# Ventajas
1. No requiere guardar trayectorias completas
2. Calcula promedios on-the-fly
3. Memoria constante independiente de N_samples
4. Fácil agregar nuevos observables

# Observables implementados
- :magnetization - Magnetización por sitio ⟨σᵢ(t)⟩
- :energy - Energía total con error estándar
- :nn_correlation - Correlación entre vecinos ⟨σᵢ σᵢ₊₁⟩
- :overlap - Overlap entre réplicas (parallel tempering)
- :all_correlations - Matriz completa ⟨σᵢ σⱼ⟩ (costoso)

# Uso típico
julia

# En la simulación
observables = [:magnetization, :energy, :nn_correlation]
result = run_monte_carlo_general(params, ..., observables=observables)

# Acceder a resultados
mag = result[:observables][:magnetization]
E = result[:observables][:energy]

# Post-procesamiento
chi = compute_susceptibility(mag, params.betas, params.N)
C = compute_specific_heat(result[:observables][:energy_variance], params.betas)


# Agregar nuevos observables
1. Agregar caso en initialize_observable_accumulators
2. Agregar caso en accumulate_observables!
3. Agregar caso en finalize_observables
"""