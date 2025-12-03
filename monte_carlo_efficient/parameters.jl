# ============================================
# ESTRUCTURA DE PARÁMETROS
# ============================================

"""
    MCParameters

Parámetros para simulaciones Monte Carlo en cadenas de Ising.
Estructura mutable que permite modificar parámetros sin recrear todo.

# Campos físicos
- `betas::Vector{Float64}`: Temperaturas inversas para cada cadena
- `j_vector::Vector{Float64}`: Acoplamientos entre vecinos (longitud N-1)
- `h_vector::Vector{Float64}`: Campos magnéticos locales (longitud N)

# Campos de dinámica
- `p0::Float64`: Probabilidad de mantener el espín en Glauber (default: 0.0)
- `s::Float64`: Tasa de swap fijo para parallel tempering (default: 0.1)

# Campos de simulación
- `seed::Int`: Semilla para reproducibilidad (default: 123)

# Constructor flexible
El constructor acepta `betas` como Float64 (una cadena) o Vector (múltiples cadenas).

# Ejemplos
```julia
# Una cadena
params = MCParameters(
    beta = 1.0,
    j_vector = ones(9),
    h_vector = zeros(10)
)

# Dos cadenas con swap
params = MCParameters(
    betas = [0.5, 1.5],
    j_vector = ones(9),
    h_vector = zeros(10),
    s = 0.2
)

# Modificar después
params.p0 = 0.1
params.betas[1] = 2.0
```
"""
mutable struct MCParameters
    # Parámetros físicos
    N::Int
    betas::Vector{Float64}
    j_vector::Vector{Float64}
    h_vector::Vector{Float64}
    
    # Parámetros de dinámica
    p0::Float64
    s::Float64
    
    # Parámetros de simulación
    seed::Int
    
    # Constructor interno completo
    function MCParameters(betas, j_vector, h_vector, p0, s, seed)
        @assert length(j_vector) == length(h_vector) - 1 "j_vector debe tener longitud N-1"
        @assert 0 <= p0 <= 1 "p0 debe estar en [0,1]"
        @assert s >= 0 "s debe ser no negativo"
        new(betas, j_vector, h_vector, p0, s, seed)
    end
end

# Constructor externo flexible con keywords
"""
    MCParameters(; beta=nothing, betas=nothing, j_vector, h_vector, 
                  p0=0.0, s=0.1, seed=123)

Constructor flexible que acepta tanto `beta` (Float) como `betas` (Vector).
"""
function MCParameters(;
    beta::Union{Nothing, Real} = nothing,
    betas::Union{Nothing, Vector{<:Real}} = nothing,
    j_vector::Vector{<:Real},
    h_vector::Vector{<:Real},
    p0::Real = 0.0,
    s::Real = 0.0,
    seed::Int = 123
)
    # Manejo flexible de beta/betas
    if beta !== nothing && betas !== nothing
        error("Especifica solo 'beta' o 'betas', no ambos")
    elseif beta !== nothing
        betas_vec = [Float64(beta)]
    elseif betas !== nothing
        betas_vec = Float64.(betas)
    else
        error("Debes especificar 'beta' o 'betas'")
    end
    
    return MCParameters(
        betas_vec,
        Float64.(j_vector),
        Float64.(h_vector),
        Float64(p0),
        Float64(s),
        seed
    )
end

# ============================================
# FUNCIONES DE UTILIDAD
# ============================================

"""
    n_chains(params::MCParameters)

Retorna el número de cadenas (longitud del vector de betas).
"""
n_chains(params::MCParameters) = length(params.betas)

"""
    N(params::MCParameters)

Retorna el número de espines (longitud de h_vector).
"""
Base.length(params::MCParameters) = length(params.h_vector)
N(params::MCParameters) = length(params.h_vector)

"""
    validate(params::MCParameters)

Valida que los parámetros sean consistentes.
"""
function validate(params::MCParameters)
    N = length(params.h_vector)
    @assert length(params.j_vector) == N - 1 "j_vector debe tener longitud N-1"
    @assert 0 <= params.p0 <= 1 "p0 debe estar en [0,1]"
    @assert params.s >= 0 "s debe ser no negativo"
    return true
end

# ============================================
# FUNCIONES DE DISPLAY
# ============================================

function Base.show(io::IO, params::MCParameters)
    n_ch = n_chains(params)
    N_val = N(params)
    println(io, "MCParameters:")
    println(io, "  N (espines): $N_val")
    println(io, "  Cadenas: $n_ch")
    println(io, "  β: $(params.betas)")
    println(io, "  p₀: $(params.p0)")
    if n_ch > 1
        println(io, "  s (swap rate): $(params.s)")
    end
    println(io, "  seed: $(params.seed)")
end

# ============================================
# EJEMPLOS DE USO
# ============================================

"""
Ejemplos de creación de parámetros:

# Una cadena homogénea
params = MCParameters(
    beta = 1.0,
    j_vector = ones(9),
    h_vector = zeros(10)
)

# Una cadena con campo inhomogéneo
params = MCParameters(
    beta = 2.0,
    j_vector = ones(9),
    h_vector = [0.1 * i for i in 1:10],
    p0 = 0.05
)

# Dos cadenas para parallel tempering
params = MCParameters(
    betas = [0.5, 2.0],
    j_vector = ones(19),
    h_vector = zeros(20),
    s = 0.15,
)

# Modificar después de crear
params.betas[1] = 1.0  # Cambiar temperatura de cadena 1
params.p0 = 0.1        # Cambiar probabilidad de no flip
"""