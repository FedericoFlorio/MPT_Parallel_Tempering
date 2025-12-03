# ============================================
# REGLAS DE ACTUALIZACIÓN (UPDATE RULES)
# ============================================

# ============================================
# FUNCIONES AUXILIARES
# ============================================

"""
    glauber_transition_rate(sigma_neighbors, sigma_new, site_index, beta, j_vector, h_vector, p0)

Calcula la probabilidad de transición de Glauber P(σᵢᵗ⁺¹ = sigma_new | configuración actual).

# Argumentos
- `sigma_neighbors`: Vecinos del sitio (ver formato abajo)
- `sigma_new`: Nuevo valor del espín (+1 o -1)
- `site_index`: Índice del sitio (1 a N)
- `beta`: Temperatura inversa
- `j_vector`: Acoplamientos (longitud N-1)
- `h_vector`: Campos magnéticos (longitud N)
- `p0`: Probabilidad de mantener el espín sin cambio

# Formato de sigma_neighbors
- Sitio 1: [σ₁, σ₂] (espín actual y vecino derecho)
- Sitio N: [σₙ₋₁, σₙ] (vecino izquierdo y espín actual)
- Sitio intermedio: [σᵢ₋₁, σᵢ, σᵢ₊₁] (vecino izq, actual, vecino der)
"""
function glauber_transition_rate(sigma_neighbors, sigma_new, site_index, beta, 
                                j_vector, h_vector, p0)
    N = length(h_vector)
    
    if site_index == 1
        # Sitio 1: solo vecino derecho
        h_eff = j_vector[1] * sigma_neighbors[2] + h_vector[1]
        sigma_current = sigma_neighbors[1]
        
    elseif site_index == N
        # Sitio N: solo vecino izquierdo
        h_eff = j_vector[end] * sigma_neighbors[1] + h_vector[end]
        sigma_current = sigma_neighbors[2]
        
    else
        # Sitio intermedio
        h_eff = j_vector[site_index - 1] * sigma_neighbors[1] + 
                j_vector[site_index] * sigma_neighbors[3] + 
                h_vector[site_index]
        sigma_current = sigma_neighbors[2]
    end
    
    # Dinámica de Glauber: P(σ → σ') ∝ exp(β σ' h_eff)
    glauber_prob = exp(beta * sigma_new * h_eff) / (2 * cosh(beta * h_eff))
    
    # Agregar componente de mantener el estado (si p0 > 0)
    if sigma_new == sigma_current
        return (1 - p0) * glauber_prob + p0
    else
        return (1 - p0) * glauber_prob
    end
end

"""
    compute_local_energy_change(spins, site_index, j_vector, h_vector)

Calcula ΔE si se flippea el espín en `site_index`.
ΔE = E(flip) - E(current) = -2 σᵢ hᵢᵉᶠᶠ

donde hᵢᵉᶠᶠ = Jᵢ₋₁ σᵢ₋₁ + Jᵢ σᵢ₊₁ + hᵢ
"""
function compute_local_energy_change(spins, site_index, j_vector, h_vector)
    N = length(spins)
    sigma_i = spins[site_index]
    
    # Campo efectivo
    h_eff = h_vector[site_index]
    
    if site_index > 1
        h_eff += j_vector[site_index - 1] * spins[site_index - 1]
    end
    
    if site_index < N
        h_eff += j_vector[site_index] * spins[site_index + 1]
    end
    
    # ΔE = -2 σᵢ hᵢᵉᶠᶠ
    return -2 * sigma_i * h_eff
end

# ============================================
# UPDATE PARALELO (GLAUBER)
# ============================================

"""
    parallel_update!(spins_new, spins, beta, j_vector, h_vector, p0, rng)

Actualización paralela tipo Glauber: todos los espines se actualizan simultáneamente.

# Argumentos
- `spins_new::Vector{Int}`: Vector donde se almacena la nueva configuración
- `spins::Vector{Int}`: Configuración actual
- `beta::Float64`: Temperatura inversa
- `j_vector::Vector{Float64}`: Acoplamientos
- `h_vector::Vector{Float64}`: Campos magnéticos
- `p0::Float64`: Probabilidad de no cambio (default: 0.0)
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `spins_new` con la nueva configuración

# Nota
Después de llamar esta función, típicamente haces:
`spins, spins_new = spins_new, spins` para swap eficiente.
"""
function parallel_update!(spins_new, spins, beta, j_vector, h_vector, p0, rng)
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
        p_up = glauber_transition_rate(sigma_neighbors, 1, i, beta, j_vector, h_vector, p0)
        p_down = glauber_transition_rate(sigma_neighbors, -1, i, beta, j_vector, h_vector, p0)
        
        # Normalizar (por seguridad numérica)
        p_total = p_up + p_down
        p_up /= p_total
        
        # Muestrear nuevo estado
        spins_new[i] = rand(rng) < p_up ? 1 : -1
    end
end

# ============================================
# UPDATE SECUENCIAL (GLAUBER)
# ============================================

"""
    sequential_update!(spins, beta, j_vector, h_vector, p0, rng)

Actualización secuencial tipo Glauber: se escoge UN espín al azar y solo ese se actualiza.

# Argumentos
- `spins::Vector{Int}`: Configuración actual (se modifica in-place)
- `beta::Float64`: Temperatura inversa
- `j_vector::Vector{Float64}`: Acoplamientos
- `h_vector::Vector{Float64}`: Campos magnéticos
- `p0::Float64`: Probabilidad de no cambio (default: 0.0)
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `spins` actualizando un único espín escogido aleatoriamente

# Nota
En cada paso temporal se actualiza UN solo espín. Para comparar con parallel_update,
considera usar T_sequential ≈ N × T_parallel, ya que en promedio cada espín es 
actualizado una vez cada N pasos secuenciales.
"""
function sequential_update!(spins, beta, j_vector, h_vector, p0, rng)
    N = length(spins)
    
    # Escoger un sitio al azar
    i = rand(rng, 1:N)
    
    # Preparar vecinos según la posición
    if i == 1
        sigma_neighbors = [spins[1], spins[2]]
    elseif i == N
        sigma_neighbors = [spins[N-1], spins[N]]
    else
        sigma_neighbors = [spins[i-1], spins[i], spins[i+1]]
    end
    
    # Calcular probabilidades para σᵢ = +1 y σᵢ = -1
    # Nota: forzamos p0=0 en Glauber secuencial (puedes cambiarlo si quieres)
    p_up = glauber_transition_rate(sigma_neighbors, 1, i, beta, j_vector, h_vector, 0.0)
    p_down = glauber_transition_rate(sigma_neighbors, -1, i, beta, j_vector, h_vector, 0.0)
    
    # Normalizar
    p_total = p_up + p_down
    p_up /= p_total
    
    # Muestrear y actualizar el espín escogido
    spins[i] = rand(rng) < p_up ? 1 : -1
end

# ============================================
# UPDATE METROPOLIS
# ============================================

"""
    metropolis_update!(spins, beta, j_vector, h_vector, rng)

Actualización tipo Metropolis: se escoge UN espín al azar y se decide si flippearlo
usando el criterio de Metropolis.

# Criterio de Metropolis
1. Se escoge un espín i al azar
2. Se calcula ΔE = E(flip) - E(current)
3. Si ΔE ≤ 0: se acepta el flip automáticamente
4. Si ΔE > 0: se acepta con probabilidad exp(-β ΔE)

# Argumentos
- `spins::Vector{Int}`: Configuración actual (se modifica in-place)
- `beta::Float64`: Temperatura inversa
- `j_vector::Vector{Float64}`: Acoplamientos
- `h_vector::Vector{Float64}`: Campos magnéticos
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `spins` flippeando potencialmente un espín según el criterio de Metropolis

# Retorna
- `accepted::Bool`: true si se aceptó el flip, false si no

# Nota
Esta es una dinámica secuencial: en cada paso se intenta flippear UN solo espín.
Para comparar con parallel_update, usa T_metropolis ≈ N × T_parallel.
"""
function metropolis_update!(spins, beta, j_vector, h_vector, rng)
    N = length(spins)
    
    # Escoger un sitio al azar
    i = rand(rng, 1:N)
    
    # Calcular cambio de energía si se flippea el espín i
    delta_E = compute_local_energy_change(spins, i, j_vector, h_vector)
    
    # Criterio de Metropolis
    accepted = false
    if delta_E <= 0
        # Disminuye energía → aceptar automáticamente
        spins[i] = -spins[i]
        accepted = true
    else
        # Aumenta energía → aceptar con probabilidad exp(-β ΔE)
        if rand(rng) < exp(-beta * delta_E)
            spins[i] = -spins[i]
            accepted = true
        end
    end
    
    return accepted
end

# ============================================
# FUNCIONES DE ALTO NIVEL
# ============================================

"""
    apply_update!(spins, spins_new, update_rule, beta, j_vector, h_vector, p0, rng)

Aplica la regla de actualización especificada.

# Argumentos
- `update_rule::Symbol`: `:parallel`, `:sequential`, o `:metropolis`
- `spins_new`: Solo usado para `:parallel` (puede ser `nothing` para otros)

# Retorna
- `accepted::Union{Bool, Nothing}`: Solo para Metropolis, indica si se aceptó el flip
"""
function apply_update!(spins, spins_new, update_rule::Symbol, beta, 
                      j_vector, h_vector, p0, rng)
    if update_rule == :parallel
        parallel_update!(spins_new, spins, beta, j_vector, h_vector, p0, rng)
        return nothing
    elseif update_rule == :sequential
        sequential_update!(spins, beta, j_vector, h_vector, p0, rng)
        return nothing
    elseif update_rule == :metropolis
        accepted = metropolis_update!(spins, beta, j_vector, h_vector, rng)
        return accepted
    else
        error("update_rule desconocida: $update_rule. Opciones: :parallel, :sequential, :metropolis")
    end
end

# ============================================
# INFORMACIÓN Y DEBUGGING
# ============================================

"""
    estimate_acceptance_rate(spins, beta, j_vector, h_vector, n_steps, rng)

Estima la tasa de aceptación de Metropolis para una configuración dada.
Útil para diagnosticar si la temperatura es apropiada.
"""
function estimate_acceptance_rate(spins, beta, j_vector, h_vector, n_steps::Int, rng)
    spins_copy = copy(spins)
    acceptances = 0
    
    for _ in 1:n_steps
        if metropolis_update!(spins_copy, beta, j_vector, h_vector, rng)
            acceptances += 1
        end
    end
    
    return acceptances / n_steps
end

"""
Información sobre las dinámicas implementadas:

# Parallel Update (Glauber)
- Todos los espines se actualizan simultáneamente
- Usa las configuraciones del tiempo t para calcular probabilidades
- Un "paso temporal" = una actualización de TODOS los espines
- Más rápida computacionalmente
- No satisface balance detallado en general

# Sequential Update (Glauber)
- Un espín aleatorio se actualiza por paso
- Balance detallado se satisface
- T_sequential ≈ N × T_parallel para comparación justa
- Más lenta pero más fiel a la dinámica continua

# Metropolis Update
- Un espín aleatorio se intenta flippear por paso
- Criterio: acepta si ΔE ≤ 0, sino con prob exp(-β ΔE)
- Balance detallado garantizado
- T_metropolis ≈ N × T_parallel para comparación justa
- Tasa de aceptación típica: 20-50% es óptimo

# Recomendaciones
- Para equilibrio rápido: usar :parallel
- Para muestreo correcto: usar :sequential o :metropolis
- Metropolis es más eficiente a bajas temperaturas
- Glauber secuencial puede ser mejor a altas temperaturas
"""