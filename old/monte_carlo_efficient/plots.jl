# ============================================
# FUNCIONES DE PLOTEO GENERAL
# ============================================

using Plots

# ============================================
# FUNCIÓN PRINCIPAL DE PLOTEO
# ============================================

"""
    plot_results(result; observables=nothing, kwargs...)

Función general para plotear resultados de simulaciones Monte Carlo.
Se adapta automáticamente al número de cadenas y observables disponibles.

# Argumentos
- `result::Dict`: Resultado de run_monte_carlo_general
- `observables::Union{Nothing, Vector{Symbol}}`: Observables a plotear (default: todos los disponibles)
- `kwargs...`: Argumentos adicionales para personalizar plots

# Keywords opcionales
- `title_prefix::String`: Prefijo para títulos (default: "")
- `save_path::Union{Nothing, String}`: Ruta para guardar figura (default: nothing)
- `figsize::Tuple{Int,Int}`: Tamaño de figura (default: (1000, 800))
- `chain_labels::Union{Nothing, Vector{String}}`: Etiquetas personalizadas para cadenas
- `time_units::String`: Unidades de tiempo para eje x (default: "steps")

# Retorna
- Plot de Plots.jl que puede ser mostrado o guardado

# Ejemplos
```julia
# Plotear todos los observables disponibles
plot_results(result)

# Plotear solo magnetización y energía
plot_results(result, observables=[:magnetization, :energy])

# Personalizar y guardar
plot_results(result, 
    title_prefix="Sistema de 20 espines",
    save_path="resultados.png",
    figsize=(1200, 900)
)
```
"""
function plot_results(result; 
                     observables=nothing, 
                     title_prefix="",
                     save_path=nothing,
                     figsize=(1000, 800),
                     chain_labels=nothing,
                     time_units="steps",
                     kwargs...)
    
    params = result[:params]
    obs_data = result[:observables]
    n_ch = n_chains(params)
    N = params.N
    
    # Si no se especifican observables, usar todos los disponibles
    if observables === nothing
        observables = collect(keys(obs_data))
    end
    
    # Filtrar solo observables que existen
    observables = [obs for obs in observables if haskey(obs_data, obs)]
    
    if isempty(observables)
        error("No hay observables para plotear")
    end
    
    # Determinar layout según número de observables
    n_obs = length(observables)
    n_rows = ceil(Int, sqrt(n_obs))
    n_cols = ceil(Int, n_obs / n_rows)
    
    # Crear labels para cadenas
    if chain_labels === nothing
        if n_ch == 1
            chain_labels = [""]
        else
            chain_labels = ["Chain $i (β=$(params.betas[i]))" for i in 1:n_ch]
        end
    end
    
    # Crear figura con subplots
    plots_array = []
    
    for obs in observables
        if obs == :magnetization
            p = plot_magnetization(obs_data[obs], params, chain_labels, time_units)
            
        elseif obs == :energy
            p = plot_energy(obs_data, params, chain_labels, time_units)
            
        elseif obs == :nn_correlation
            p = plot_nn_correlation(obs_data[obs], params, chain_labels, time_units)
            
        elseif obs == :overlap
            p = plot_overlap(obs_data, params, time_units)
            
        elseif obs == :all_correlations
            p = plot_correlation_matrix(obs_data[obs], params)
            
        else
            @warn "No hay función de plot para observable: $obs"
            continue
        end
        
        push!(plots_array, p)
    end
    
    # Combinar subplots
    final_plot = plot(plots_array..., 
                     layout=(n_rows, n_cols),
                     size=figsize,
                     plot_title=title_prefix)
    
    # Guardar si se especifica
    if save_path !== nothing
        savefig(final_plot, save_path)
        println("Figura guardada en: $save_path")
    end
    
    return final_plot
end

# ============================================
# FUNCIONES DE PLOTEO POR OBSERVABLE
# ============================================

"""
    plot_magnetization(mag_data, params, chain_labels, time_units)

Plotea la magnetización marginal por sitio.
"""
function plot_magnetization(mag_data, params, chain_labels, time_units)
    N = params.N
    n_ch = n_chains(params)
    
    if n_ch == 1
        # Una cadena: mag_data es (N, T)
        T = size(mag_data, 2)
        t_range = 0:(T-1)
        
        p = heatmap(t_range, 1:N, mag_data,
                   xlabel="Time ($time_units)",
                   ylabel="Site",
                   title="Magnetization ⟨σᵢ(t)⟩",
                   color=:RdBu,
                   clim=(-1, 1),
                   colorbar_title="⟨σᵢ⟩")
    else
        # Múltiples cadenas: mag_data es (n_chains, N, T)
        T = size(mag_data, 3)
        t_range = 0:(T-1)
        
        # Crear subplots para cada cadena
        plots_chains = []
        for c in 1:n_ch
            pc = heatmap(t_range, 1:N, mag_data[c, :, :],
                        xlabel="Time ($time_units)",
                        ylabel="Site",
                        title=chain_labels[c],
                        color=:RdBu,
                        clim=(-1, 1),
                        colorbar_title="⟨σᵢ⟩")
            push!(plots_chains, pc)
        end
        
        p = plot(plots_chains..., 
                layout=(1, n_ch),
                plot_title="Magnetization ⟨σᵢ(t)⟩")
    end
    
    return p
end

"""
    plot_energy(obs_data, params, chain_labels, time_units)

Plotea la energía total en función del tiempo.
"""
function plot_energy(obs_data, params, chain_labels, time_units)
    energy = obs_data[:energy]
    energy_error = obs_data[:energy_error]
    n_ch = n_chains(params)
    
    if n_ch == 1
        # Una cadena
        T = length(energy)
        t_range = 0:(T-1)
        
        p = plot(t_range, energy,
                ribbon=energy_error,
                xlabel="Time ($time_units)",
                ylabel="Energy",
                title="Total Energy ⟨E(t)⟩",
                label="E(t)",
                linewidth=2,
                fillalpha=0.3,
                legend=:best)
    else
        # Múltiples cadenas
        T = size(energy, 2)
        t_range = 0:(T-1)
        
        p = plot(xlabel="Time ($time_units)",
                ylabel="Energy",
                title="Total Energy ⟨E(t)⟩",
                legend=:best)
        
        for c in 1:n_ch
            plot!(p, t_range, energy[c, :],
                 ribbon=energy_error[c, :],
                 label=chain_labels[c],
                 linewidth=2,
                 fillalpha=0.3)
        end
    end
    
    return p
end

"""
    plot_nn_correlation(nn_corr_data, params, chain_labels, time_units)

Plotea la correlación entre vecinos cercanos.
"""
function plot_nn_correlation(nn_corr_data, params, chain_labels, time_units)
    n_ch = n_chains(params)
    
    if n_ch == 1
        # Una cadena
        T = length(nn_corr_data)
        t_range = 0:(T-1)
        
        p = plot(t_range, nn_corr_data,
                xlabel="Time ($time_units)",
                ylabel="⟨σᵢ σᵢ₊₁⟩",
                title="Nearest-Neighbor Correlation",
                label="NN Correlation",
                linewidth=2,
                legend=:best)
    else
        # Múltiples cadenas
        T = size(nn_corr_data, 2)
        t_range = 0:(T-1)
        
        p = plot(xlabel="Time ($time_units)",
                ylabel="⟨σᵢ σᵢ₊₁⟩",
                title="Nearest-Neighbor Correlation",
                legend=:best)
        
        for c in 1:n_ch
            plot!(p, t_range, nn_corr_data[c, :],
                 label=chain_labels[c],
                 linewidth=2)
        end
    end
    
    return p
end

"""
    plot_overlap(obs_data, params, time_units)

Plotea el overlap entre réplicas (solo para n_chains ≥ 2).
"""
function plot_overlap(obs_data, params, time_units)
    overlap_data = obs_data[:overlap]
    overlap_pairs = obs_data[:overlap_pairs]
    
    n_pairs = size(overlap_data, 1)
    T = size(overlap_data, 2)
    t_range = 0:(T-1)
    
    p = plot(xlabel="Time ($time_units)",
            ylabel="Overlap Q(t)",
            title="Replica Overlap",
            legend=:best)
    
    for i in 1:n_pairs
        pair = overlap_pairs[i]
        label_str = "Q($(pair[1]),$(pair[2]))"
        plot!(p, t_range, overlap_data[i, :],
             label=label_str,
             linewidth=2)
    end
    
    return p
end

"""
    plot_correlation_matrix(corr_data, params)

Plotea la matriz de correlaciones completa al tiempo final.
"""
function plot_correlation_matrix(corr_data, params)
    N = params.N
    n_ch = n_chains(params)
    
    if n_ch == 1
        # Una cadena: corr_data es (N, N, T)
        T = size(corr_data, 3)
        C_final = corr_data[:, :, T]
        
        p = heatmap(1:N, 1:N, C_final,
                   xlabel="Site j",
                   ylabel="Site i",
                   title="Correlation Matrix ⟨σᵢ σⱼ⟩ (t=$T)",
                   color=:RdBu,
                   clim=(-1, 1),
                   aspect_ratio=:equal,
                   colorbar_title="⟨σᵢ σⱼ⟩")
    else
        # Múltiples cadenas
        T = size(corr_data, 4)
        
        plots_chains = []
        for c in 1:n_ch
            C_final = corr_data[c, :, :, T]
            pc = heatmap(1:N, 1:N, C_final,
                        xlabel="Site j",
                        ylabel="Site i",
                        title="Chain $c (β=$(params.betas[c]))",
                        color=:RdBu,
                        clim=(-1, 1),
                        aspect_ratio=:equal,
                        colorbar_title="⟨σᵢ σⱼ⟩")
            push!(plots_chains, pc)
        end
        
        p = plot(plots_chains...,
                layout=(1, n_ch),
                plot_title="Correlation Matrix ⟨σᵢ σⱼ⟩ (t=$T)")
    end
    
    return p
end

# ============================================
# PLOTS ADICIONALES Y DIAGNÓSTICO
# ============================================

"""
    plot_magnetization_profile(result; t=nothing)

Plotea el perfil de magnetización ⟨σᵢ⟩ vs sitio en un tiempo específico.

# Argumentos
- `t::Union{Nothing, Int}`: Tiempo a plotear (default: último tiempo)
"""
function plot_magnetization_profile(result; t=nothing)
    obs_data = result[:observables]
    params = result[:params]
    mag_data = obs_data[:magnetization]
    N = params.N
    n_ch = n_chains(params)
    
    if t === nothing
        if n_ch == 1
            t = size(mag_data, 2)
        else
            t = size(mag_data, 3)
        end
    end
    
    p = plot(xlabel="Site",
            ylabel="⟨σᵢ⟩",
            title="Magnetization Profile at t=$t",
            legend=:best,
            ylim=(-1.1, 1.1))
    
    if n_ch == 1
        plot!(p, 1:N, mag_data[:, t],
             label="Magnetization",
             linewidth=2,
             marker=:circle)
    else
        for c in 1:n_ch
            plot!(p, 1:N, mag_data[c, :, t],
                 label="Chain $c (β=$(params.betas[c]))",
                 linewidth=2,
                 marker=:circle)
        end
    end
    
    # Línea en cero
    hline!(p, [0], linestyle=:dash, color=:black, label="", alpha=0.3)
    
    return p
end

"""
    plot_energy_histogram(result; t=nothing, bins=30)

Plotea histograma de energías (requiere trayectorias guardadas).
"""
function plot_energy_histogram(result; t=nothing, bins=30)
    if !haskey(result, :trajectories)
        error("Se necesitan trayectorias guardadas para este plot. Usa save_trajectory=true")
    end
    
    params = result[:params]
    trajectories = result[:trajectories]
    n_ch = n_chains(params)
    
    if t === nothing
        t = size(trajectories[1], 3)
    end
    
    p = plot(xlabel="Energy",
            ylabel="Frequency",
            title="Energy Distribution at t=$t",
            legend=:best)
    
    for c in 1:n_ch
        N_samples = size(trajectories[c], 1)
        energies = zeros(N_samples)
        
        for sample in 1:N_samples
            spins = trajectories[c][sample, :, t]
            energies[sample] = compute_total_energy(spins, params.j_vector, params.h_vector)
        end
        
        histogram!(p, energies,
                  bins=bins,
                  alpha=0.6,
                  normalize=:probability,
                  label="Chain $c (β=$(params.betas[c]))")
    end
    
    return p
end

"""
    plot_swap_acceptance_diagnostic(result; window=100)

Diagnóstico de aceptación de swaps (solo si hay info de swaps guardada).
"""
function plot_swap_acceptance_diagnostic(swap_history::Vector{Bool}; window=100)
    T = length(swap_history)
    
    # Calcular tasa de aceptación en ventanas móviles
    acceptance_rate = zeros(T - window + 1)
    
    for i in 1:(T - window + 1)
        acceptance_rate[i] = mean(swap_history[i:(i + window - 1)])
    end
    
    p = plot(window:T, acceptance_rate,
            xlabel="Time",
            ylabel="Acceptance Rate",
            title="Swap Acceptance Rate (window=$window)",
            label="Acceptance",
            linewidth=2,
            ylim=(0, 1))
    
    # Líneas de referencia
    hline!(p, [0.2, 0.4], linestyle=:dash, color=[:red, :green], 
           label=["Low (20%)" "Optimal (20-40%)"], alpha=0.5)
    
    return p
end

"""
    plot_comparison_parallel_vs_sequential(result_parallel, result_sequential)

Compara resultados de dinámicas paralela vs secuencial.
"""
function plot_comparison_parallel_vs_sequential(result_parallel, result_sequential)
    obs_par = result_parallel[:observables]
    obs_seq = result_sequential[:observables]
    
    # Extraer magnetización
    mag_par = obs_par[:magnetization]
    mag_seq = obs_seq[:magnetization]
    
    T_par = size(mag_par, 2)
    T_seq = size(mag_seq, 2)
    
    # Magnetización total
    M_par = vec(sum(mag_par, dims=1))
    M_seq = vec(sum(mag_seq, dims=1))
    
    p = plot(layout=(2, 1), size=(800, 600))
    
    # Subplot 1: Magnetización total
    plot!(p[1], 0:(T_par-1), M_par,
         label="Parallel",
         linewidth=2,
         xlabel="Time (parallel steps)",
         ylabel="Total Magnetization")
    
    # Escalar tiempo secuencial
    N = size(mag_par, 1)
    t_seq_scaled = (0:(T_seq-1)) ./ N
    plot!(p[1], t_seq_scaled, M_seq,
         label="Sequential (scaled)",
         linewidth=2)
    
    # Subplot 2: Diferencia
    # Interpolar para comparar
    M_seq_interp = [M_seq[min(end, round(Int, t*N) + 1)] for t in 0:(T_par-1)]
    
    plot!(p[2], 0:(T_par-1), abs.(M_par .- M_seq_interp),
         label="Absolute Difference",
         linewidth=2,
         xlabel="Time (parallel steps)",
         ylabel="|M_par - M_seq|")
    
    return p
end
