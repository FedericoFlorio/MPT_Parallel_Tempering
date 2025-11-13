# ============================================
# COMPROBACIONES Y DISTRIBUCIONES MARGINALES
# ============================================

"""
    sum_one_check(B): Verifica que la suma sobre todas las configuraciones de espines de la distribución 
    representada por el TensorTrain B sea 1.
"""
function sum_one_check(B)
    N = length(B.tensors)
    Q = size(B.tensors[1], 3)
    sum_tensor = fill(1.0, 1, 1)
    for i in 1:N
        Bi_sum = zeros(size(B.tensors[i], 1), size(B.tensors[i], 2))
        for q in 1:Q
            Bi_sum .+= B.tensors[i][:,:,q,1]
        end
        sum_tensor *= Bi_sum
    end
    return only(sum_tensor)
end

"""
    marginal_distribution(B, k): Calcula la distribución marginal P(σ_k) para el sitio k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_distribution(B,k)
    N = length(B.tensors)
    K = size(B.tensors[1], 3)
    left_distribution = 1
    for i in 1:(k-1)
        Bi_sum = zeros(size(B.tensors[i], 1), size(B.tensors[i], 2))
        for q in 1:K
            Bi_sum .+= B.tensors[i][:,:,q,1]
        end
        left_distribution *= Bi_sum
    end
    right_distribution = 1
    for i in (k+1):N
        Bi_sum = zeros(size(B.tensors[i], 1), size(B.tensors[i], 2))
        for q in 1:K
            Bi_sum .+= B.tensors[i][:,:,q,1]
        end
        right_distribution *= Bi_sum
    end

    distribution = [(left_distribution * B.tensors[k][:,:, q, 1] * right_distribution)[1] for q in 1:K]
    return distribution
end

"""
    marginal_distribution_system(B): Calcula la distribución marginal P(σ_k) para todos los sitios k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""

function marginal_distribution_system(B)
    N = length(B.tensors)
    distributions = []
    for k in 1:N
        push!(distributions, marginal_distribution(B, k))
    end
    return distributions
end

"""
    marginal_expected_value(B, k): Calcula el valor esperado marginal E[σ_k] para el sitio k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_expected_value_simple(B, k)
    return  marginal_distribution(B, k)[1]*(-1) + marginal_distribution(B, k)[2]*(1)
end

"""
    marginal_expected_value(B, k): Calcula el valor esperado marginal E[σ_k^x], E[σ_k^y] para el sitio k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_expected_value_parallel(B, k)
    function suma(a,b,B=B,k=k)
        return marginal_distribution(B, k)[a] + marginal_distribution(B, k)[b]
    end
    return  (suma(1,3)*(-1) + suma(2,4)*(1), suma(1,2)*(-1) + suma(3,4)*(1))
end

"""
    marginal_expected_value_system(B): Calcula el valor esperado marginal E[σ_k] para todos los sitios k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_ev_system(B)
    N = length(B.tensors)
    expected_values = zeros(N)
    for k in 1:N
        expected_values[k] = marginal_expected_value(B, k)
    end
    return expected_values
end

"""
    marginal_expected_value_system(B): Calcula el valor esperado marginal E[σ_k] para todos los sitios k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_ev_parallel_system(B)
    N = length(B.tensors)
    expected_values = []
    for k in 1:N
        push!(expected_values, marginal_expected_value_parallel(B, k))
    end
    return expected_values
end



"""
    covariance_between_chains(B)
Calcula la correlación entre las dos cadenas en cada sitio k:
Corr(σ_k^X, σ_k^Y) = E[σ_k^X σ_k^Y] - E[σ_k^X] E[σ_k^Y] 
"""

function covariance_between_chains(B)
    N = length(B.tensors)
    marginals = marginal_distribution_system(B)
    simple_ev = marginal_ev_parallel_system(B)
    correlations = [marginals[k][1]+marginals[k][4]-marginals[k][2]-marginals[k][3] - simple_ev[k][1]*simple_ev[k][2] for k in 1:N]
    return correlations
end

"""
    correlation_between_chains(B)
Calcula la correlación normalizada entre las dos cadenas en cada sitio k:
Corr(σ_k^X, σ_k^Y) = Cov(σ_k^X, σ_k^Y) / (sqrt(1 - E[σ_k^X]^2) * sqrt(1 - E[σ_k^Y]^2))
"""

function correlation_between_chains(B)
    N = length(B.tensors)
    simple_ev = marginal_ev_parallel_system(B)
    covariances = covariance_between_chains(B)
    correlations = [ covariances[k] / (sqrt(1 - simple_ev[k][1]^2) * sqrt(1 - simple_ev[k][2]^2))  for k in 1:N]
    return correlations
end