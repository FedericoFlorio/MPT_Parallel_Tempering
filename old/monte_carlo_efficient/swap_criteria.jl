# ============================================
# CRITERIOS DE SWAP PARA PARALLEL TEMPERING
# ============================================

# ============================================
# FUNCIONES AUXILIARES
# ============================================

"""
    compute_total_energy(spins, j_vector, h_vector)

Calcula la energía total de una configuración de espines.

E = -∑ᵢ Jᵢ σᵢ σᵢ₊₁ - ∑ᵢ hᵢ σᵢ

# Argumentos
- `spins::Vector{Int}`: Configuración de espines (±1)
- `j_vector::Vector{Float64}`: Acoplamientos (longitud N-1)
- `h_vector::Vector{Float64}`: Campos magnéticos (longitud N)

# Retorna
- `E::Float64`: Energía total del sistema
"""
function compute_total_energy(spins, j_vector, h_vector)
    N = length(spins)
    energy = 0.0
    
    # Término de interacción: -∑ᵢ Jᵢ σᵢ σᵢ₊₁
    for i in 1:(N-1)
        energy -= j_vector[i] * spins[i] * spins[i+1]
    end
    
    # Término de campo magnético: -∑ᵢ hᵢ σᵢ
    for i in 1:N
        energy -= h_vector[i] * spins[i]
    end
    
    return energy
end

# ============================================
# SWAP CON TASA FIJA
# ============================================

"""
    apply_fixed_rate_swap!(chains, s, rng)

Realiza swap entre cadenas adyacentes con probabilidad fija `s`.

# Descripción
Para cada par de cadenas adyacentes (i, i+1):
- Con probabilidad `s`: intercambiar las configuraciones completas
- Con probabilidad `1-s`: no hacer nada

Este es el método más simple pero NO garantiza balance detallado.
Es útil como baseline o cuando quieres control directo sobre la tasa de swap.

# Argumentos
- `chains::Vector{Vector{Int}}`: Vector de cadenas (cada una es un Vector{Int})
- `s::Float64`: Probabilidad de swap (0 ≤ s ≤ 1)
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `chains` intercambiando potencialmente configuraciones entre cadenas adyacentes

# Retorna
- `n_swaps::Int`: Número de swaps realizados

# Ejemplos
julia
# Dos cadenas
chains = [[1, -1, 1], [-1, 1, -1]]
n_swaps = apply_fixed_rate_swap!(chains, 0.2, rng)

# Tres cadenas - intenta swap (1,2) y (2,3)
chains = [chain1, chain2, chain3]
n_swaps = apply_fixed_rate_swap!(chains, 0.1, rng)


# Nota
Si tienes n_chains cadenas, se intentan (n_chains - 1) swaps en cada llamada.
"""


function apply_fixed_rate_swap!(chains, s, rng)
    n_chains = length(chains)
    n_swaps = 0
    
    # Intentar swap entre cada par adyacente
    for i in 1:(n_chains - 1)
        if rand(rng) < s
            # Intercambiar configuraciones completas
            chains[i], chains[i+1] = chains[i+1], chains[i]
            n_swaps += 1
        end
    end
    
    return n_swaps
end

# ============================================
# SWAP CON CRITERIO DE METROPOLIS
# ============================================

"""
    apply_metropolis_swap!(chains, betas, j_vector, h_vector, rng)

Realiza swap entre cadenas adyacentes usando el criterio de Metropolis.

# Descripción
Para cada par de cadenas adyacentes (i, i+1) con temperaturas βᵢ y βᵢ₊₁:

1. Calcula energías: Eᵢ (cadena i) y Eᵢ₊₁ (cadena i+1)
2. Calcula ΔE_swap = (βᵢ - βᵢ₊₁)(Eᵢ - Eᵢ₊₁)
3. Acepta swap con probabilidad: min(1, exp(-ΔE_swap))

Criterio equivalente:
- Si ΔE_swap ≤ 0: acepta automáticamente
- Si ΔE_swap > 0: acepta con probabilidad exp(-ΔE_swap)

Este método GARANTIZA balance detallado y permite muestreo correcto de la
distribución conjunta de todas las réplicas.

# Argumentos
- `chains::Vector{Vector{Int}}`: Vector de cadenas
- `betas::Vector{Float64}`: Temperaturas inversas (una por cadena)
- `j_vector::Vector{Float64}`: Acoplamientos
- `h_vector::Vector{Float64}`: Campos magnéticos
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `chains` intercambiando potencialmente configuraciones según Metropolis

# Retorna
- `(n_attempts, n_accepted)::Tuple{Int, Int}`: Número de intentos y aceptaciones

# Ejemplos
julia
# Dos cadenas con diferentes temperaturas
chains = [chain1, chain2]
betas = [0.5, 2.0]  # Alta y baja temperatura
attempts, accepted = apply_metropolis_swap!(chains, betas, j_vec, h_vec, rng)
acceptance_rate = accepted / attempts


# Recomendaciones
- Ordena las cadenas por temperatura: β₁ < β₂ < ... < βₙ
- Típicamente βᵢ₊₁/βᵢ ≈ 1.2 - 1.5 para buena eficiencia
- Tasa de aceptación óptima: 20-40%
- Si aceptación muy baja: acerca las temperaturas
- Si aceptación muy alta: separa más las temperaturas
"""
function apply_metropolis_swap!(chains, betas, j_vector, h_vector, rng)
    n_chains = length(chains)
    n_attempts = 0
    n_accepted = 0
    
    # Intentar swap entre cada par adyacente
    for i in 1:(n_chains - 1)
        n_attempts += 1
        
        # Calcular energías de ambas cadenas
        E_i = compute_total_energy(chains[i], j_vector, h_vector)
        E_i_plus_1 = compute_total_energy(chains[i+1], j_vector, h_vector)
        
        # Calcular cambio en la función objetivo del swap
        # ΔE_swap = (βᵢ - βᵢ₊₁)(Eᵢ - Eᵢ₊₁)
        delta_E_swap = (betas[i] - betas[i+1]) * (E_i - E_i_plus_1)
        
        # Criterio de Metropolis para el swap
        if delta_E_swap <= 0
            # Swap favorable → aceptar automáticamente
            chains[i], chains[i+1] = chains[i+1], chains[i]
            n_accepted += 1
        else
            # Swap desfavorable → aceptar con probabilidad exp(-ΔE_swap)
            if rand(rng) < exp(-delta_E_swap)
                chains[i], chains[i+1] = chains[i+1], chains[i]
                n_accepted += 1
            end
        end
    end
    
    return (n_attempts, n_accepted)
end



# AGREGAR después de apply_metropolis_swap!
"""
    apply_metropolis_swap_with_energies!(chains, energies, betas, rng)

Versión eficiente que usa energías precalculadas.
"""
function apply_metropolis_swap_with_energies!(chains, energies, betas, rng)
    n_chains = length(chains)
    n_attempts = 0
    n_accepted = 0
    
    for i in 1:(n_chains - 1)
        n_attempts += 1
        
        # Usar energías ya calculadas (O(1))
        E_i = energies[i]
        E_i_plus_1 = energies[i+1]
        
        delta_E_swap = (betas[i] - betas[i+1]) * (E_i - E_i_plus_1)
        
        if delta_E_swap <= 0 || rand(rng) < exp(-delta_E_swap)
            # Swap configuraciones
            chains[i], chains[i+1] = chains[i+1], chains[i]
            # IMPORTANTE: Swap también las energías
            energies[i], energies[i+1] = energies[i+1], energies[i]
            n_accepted += 1
        end
    end
    
    return (n_attempts, n_accepted)
end
























# ============================================
# VARIANTES DE SWAP
# ============================================

"""
    apply_random_pair_swap!(chains, betas, j_vector, h_vector, rng; 
                           method=:metropolis, s=0.1)

Intenta swap entre UN par de cadenas escogidas aleatoriamente (no necesariamente adyacentes).

# Argumentos
- `method::Symbol`: `:metropolis` o `:fixed_rate`
- Para otros argumentos, ver funciones individuales

# Retorna
- `accepted::Bool`: true si se aceptó el swap

# Nota
Esta variante puede ser útil cuando tienes muchas cadenas y quieres
explorar swaps no-locales en el espacio de temperaturas.
"""

"""
function apply_random_pair_swap!(chains, betas, j_vector, h_vector, rng; 
                                method::Symbol=:metropolis, s::Float64=0.1)
    n_chains = length(chains)
    
    if n_chains < 2
        return false
    end
    
    # Escoger dos cadenas distintas al azar
    i = rand(rng, 1:n_chains)
    j = rand(rng, 1:n_chains)
    while j == i
        j = rand(rng, 1:n_chains)
    end
    
    # Asegurar i < j
    if i > j
        i, j = j, i
    end
    
    # Aplicar criterio de swap
    if method == :fixed_rate
        if rand(rng) < s
            chains[i], chains[j] = chains[j], chains[i]
            return true
        end
        return false
        
    else  # :metropolis
        E_i = compute_total_energy(chains[i], j_vector, h_vector)
        E_j = compute_total_energy(chains[j], j_vector, h_vector)
        
        delta_E_swap = (betas[i] - betas[j]) * (E_i - E_j)
        
        if delta_E_swap <= 0 || rand(rng) < exp(-delta_E_swap)
            chains[i], chains[j] = chains[j], chains[i]
            return true
        end
        return false
    end
end
"""






# ============================================
# DIAGNÓSTICOS Y UTILIDADES
# ============================================

"""
    compute_swap_acceptance_rate(chains, betas, j_vector, h_vector, 
                                 n_trials, rng)

Estima la tasa de aceptación de swaps Metropolis entre cadenas adyacentes.
Útil para optimizar la elección de temperaturas.

# Retorna
- `Vector{Float64}`: Tasa de aceptación para cada par adyacente
"""

function compute_swap_acceptance_rate(chains, betas, j_vector, h_vector, 
                                     n_trials::Int, rng)
    n_chains = length(chains)
    
    if n_chains < 2
        return Float64[]
    end
    
    # Copiar cadenas para no modificar originales
    chains_copy = [copy(chain) for chain in chains]
    
    # Acumuladores de aceptación por par
    acceptances = zeros(Int, n_chains - 1)
    
    for _ in 1:n_trials
        for i in 1:(n_chains - 1)
            E_i = compute_total_energy(chains_copy[i], j_vector, h_vector)
            E_i_plus_1 = compute_total_energy(chains_copy[i+1], j_vector, h_vector)
            
            delta_E_swap = (betas[i] - betas[i+1]) * (E_i - E_i_plus_1)
            
            if delta_E_swap <= 0 || rand(rng) < exp(-delta_E_swap)
                chains_copy[i], chains_copy[i+1] = chains_copy[i+1], chains_copy[i]
                acceptances[i] += 1
            end
        end
    end
    
    return acceptances ./ n_trials
end


"""
    suggest_beta_schedule(beta_min, beta_max, n_chains; 
                         target_acceptance=0.3, distribution=:geometric)

Sugiere un schedule de temperaturas para parallel tempering.

# Argumentos
- `beta_min::Float64`: Temperatura inversa mínima (alta temperatura)
- `beta_max::Float64`: Temperatura inversa máxima (baja temperatura)
- `n_chains::Int`: Número de cadenas
- `target_acceptance::Float64`: Tasa de aceptación objetivo (default: 0.3)
- `distribution::Symbol`: `:geometric` o `:linear`

# Retorna
- `Vector{Float64}`: Vector de betas sugerido

# Ejemplos
julia
betas = suggest_beta_schedule(0.5, 2.0, 5)
# Retorna algo como [0.5, 0.707, 1.0, 1.414, 2.0]
"""

function suggest_beta_schedule(beta_min::Float64, beta_max::Float64, n_chains::Int; 
                              target_acceptance::Float64=0.3, 
                              distribution::Symbol=:geometric)
    @assert beta_min < beta_max "beta_min debe ser menor que beta_max"
    @assert n_chains >= 2 "Se necesitan al menos 2 cadenas"
    @assert 0 < target_acceptance < 1 "target_acceptance debe estar en (0,1)"
    
    if distribution == :geometric
        # Distribución geométrica: βᵢ₊₁/βᵢ = constante
        ratio = (beta_max / beta_min)^(1/(n_chains - 1))
        return [beta_min * ratio^(i-1) for i in 1:n_chains]
        
    elseif distribution == :linear
        # Distribución lineal
        return range(beta_min, beta_max, length=n_chains) |> collect
        
    else
        error("distribution debe ser :geometric o :linear")
    end
end



# ============================================
# INFORMACIÓN
# ============================================

"""
Información sobre criterios de swap:

# Fixed Rate Swap
- Swap con probabilidad fija s
- Simple y rápido
- NO garantiza balance detallado
- Útil como baseline

# Metropolis Swap
- Usa criterio de aceptación energético
- Garantiza balance detallado
- Permite muestreo correcto
- Más lento pero riguroso

# Recomendaciones para Parallel Tempering
1. Usa Metropolis swap para resultados correctos
2. Ordena temperaturas: β₁ < β₂ < ... < βₙ
3. Ratio típico: βᵢ₊₁/βᵢ ≈ 1.2 - 1.5
4. Tasa de aceptación objetivo: 20-40%
5. Monitorea el "round-trip time": tiempo que tarda una configuración
   en ir de la temperatura más baja a la más alta y volver

# Diagnóstico
- Si aceptación < 10%: temperaturas muy separadas
- Si aceptación > 60%: temperaturas muy cercanas (ineficiente)
- Usa compute_swap_acceptance_rate() para verificar
"""