import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# Helper Functions (f, g, fprime)
# ==============================================================================

def g_fun(b, D):
    # g(b) = D / (2^b - 1)^2
    # Singularity prevention near b=0
    eps_b = 1e-6
    b_eff = np.maximum(b, eps_b)
    x = 2.0 ** b_eff
    g = D / ((x - 1.0) ** 2)
    return g

def f_fun(b, p_i, D):
    # f_i(b) = (1 - p_i) / (g(b) + p_i)
    g = g_fun(b, D)
    f = (1.0 - p_i) / (g + p_i)
    return f

def fprime_fun(b, p_i, D):
    # f_i'(b) derivative
    eps_b = 1e-6
    b_eff = np.maximum(b, eps_b)
    x = 2.0 ** b_eff
    g = D / ((x - 1.0) ** 2)
    # g'(b) = -2*D*log(2)*2^b / (2^b - 1)^3
    gprime = -2.0 * D * np.log(2.0) * x / ((x - 1.0) ** 3)
    fp = -(1.0 - p_i) * gprime / ((g + p_i) ** 2)
    return fp

def f_fun_safe(b, p_i, D):
    # Safe version for local search / waterfilling
    if b <= 0:
        return 0.0
    denom_term = (2.0 ** b - 1.0) ** 2
    if denom_term == 0:
        return 0.0
    else:
        g_val = D / denom_term
        return (1.0 - p_i) / (g_val + p_i)

def objective_bit_allocation(b, p, D):
    F = 0.0
    for i in range(len(p)):
        F += f_fun(b[i], p[i], D)
    return F

# ==============================================================================
# Algorithms
# ==============================================================================

# 1) Dynamic Programming
def solve_dp(Btot, p, D):
    K = len(p)
    # V[i][b] = max value using first i users (0 to i-1), total bits = b
    # Dimensions: (K+1) x (Btot+1)
    V = np.full((K + 1, Btot + 1), -np.inf)
    choice = np.zeros((K + 1, Btot + 1), dtype=int)
    
    # Precompute f_table for speed
    f_table = np.zeros((K, Btot + 1))
    for i in range(K):
        for b in range(Btot + 1):
            f_table[i, b] = f_fun(b, p[i], D)

    # Base case: 0 workers, 0 bits -> value 0
    V[0, 0] = 0.0
    
    for i in range(1, K + 1):  # i workers (1..K)
        # current worker index in p is i-1
        u_idx = i - 1
        for b in range(Btot + 1):
            best_val = -np.inf
            best_t = 0
            # t = bits assigned to current worker
            for t in range(b + 1):
                val_prev = V[i - 1, b - t]
                if val_prev == -np.inf:
                    continue
                val = val_prev + f_table[u_idx, t]
                if val > best_val:
                    best_val = val
                    best_t = t
            V[i, b] = best_val
            choice[i, b] = best_t
            
    F_opt = V[K, Btot]
    
    # Backtracking (optional, purely for completeness)
    b_opt = np.zeros(K)
    rem = Btot
    for i in range(K, 0, -1):
        t = choice[i, rem]
        b_opt[i-1] = t
        rem -= t
        
    return F_opt, b_opt

# 2) Greedy
def solve_greedy(Btot, p, D, F_zero, F_opt=None):
    K = len(p)
    
    # F_zero check
    if F_zero is None:
        F_zero = 0.0
        for i in range(K):
            F_zero += f_fun(0, p[i], D)
            
    # Calculate singleton values (F({v}))
    F_single = np.zeros(K)
    for i in range(K):
        # b = e_i
        F_tmp = 0.0
        for j in range(K):
            val = f_fun(1, p[j], D) if j == i else f_fun(0, p[j], D)
            F_tmp += val
        F_single[i] = F_tmp
        
    b = np.zeros(K)
    F_current = F_zero
    
    # For bounds calculation
    gamma_list = []
    alpha_list = []
    
    for t in range(Btot):
        gains = np.zeros(K)
        F_candidate = np.zeros(K)
        
        for i in range(K):
            f_now = f_fun(b[i], p[i], D)
            f_next = f_fun(b[i] + 1, p[i], D)
            gains[i] = f_next - f_now
            F_candidate[i] = F_current + gains[i]
            
            # Gamma/Alpha calculation logic
            delta_Cv = gains[i]
            delta_v = F_single[i] - F_zero
            
            if delta_v > 0 and delta_Cv > 0:
                gamma_cand = delta_v / delta_Cv
                gamma_list.append(gamma_cand)
                
                alpha_cand = 1.0 - (delta_Cv / delta_v)
                alpha_list.append(alpha_cand)
        
        # Select best
        idx = np.argmax(gains)
        b[idx] += 1
        F_current = F_candidate[idx]
        
    F_greedy = objective_bit_allocation(b, p, D)
    
    # Theoretical bounds
    gamma_greedy = 0.0
    if gamma_list:
        gamma_greedy = min(gamma_list)
        gamma_greedy = min(1.0, max(0.0, gamma_greedy))
        
    alpha_greedy = 1.0
    if alpha_list:
        alpha_greedy = max(alpha_list)
        alpha_greedy = min(1.0, max(0.0, alpha_greedy))
        
    approx_factor = np.inf
    empirical_ratio = np.nan
    
    if F_opt is not None:
        if gamma_greedy > 0 and alpha_greedy < 1:
            approx_factor = 1.0 / (gamma_greedy * (1.0 - alpha_greedy))
        
        num = F_zero - F_greedy
        denom = F_zero - F_opt
        if denom != 0:
            empirical_ratio = num / denom
            
    return F_greedy, b, empirical_ratio

# 3) Continuous Relaxation (fmincon equivalent)
def solve_continuous(Btot, p, D):
    K = len(p)
    b0 = np.ones(K) * (Btot / K)
    
    # Minimize negative Objective
    def obj_func(b):
        F = 0.0
        grad = np.zeros(K)
        for i in range(K):
            F += f_fun(b[i], p[i], D)
            grad[i] = fprime_fun(b[i], p[i], D)
        return -F, -grad
    
    # Constraints: sum(b) = Btot
    constraints = ({'type': 'eq', 'fun': lambda b: np.sum(b) - Btot})
    # Bounds: b >= 0
    bounds = [(0, None) for _ in range(K)]
    
    res = minimize(obj_func, b0, method='SLSQP', jac=True, 
                   bounds=bounds, constraints=constraints, 
                   options={'disp': False})
    
    b_cont = res.x
    F_cont = -res.fun
    return F_cont, b_cont

# 4) Top-K (Standard)
def solve_topk(Btot, p, D):
    K = len(p)
    # Sort users (ascending p is better)
    sorted_idx = np.argsort(p) # argsort gives indices
    
    max_obj = -np.inf
    best_b = np.zeros(K)
    
    for k in range(1, K + 1):
        if Btot < k:
            active_count = Btot
        else:
            active_count = k
            
        current_b = np.zeros(K)
        base_bits = Btot // active_count
        rem_bits = Btot % active_count
        
        active_users = sorted_idx[:active_count]
        
        for i in range(active_count):
            u_idx = active_users[i]
            extra = 1 if i < rem_bits else 0
            current_b[u_idx] = base_bits + extra
            
        current_obj = objective_bit_allocation(current_b, p, D)
        
        if current_obj > max_obj:
            max_obj = current_obj
            best_b = current_b.copy()
            
    return max_obj, best_b


def solve_equal(Btot, p, D):
    K = len(p)
    # Sort users (ascending p is better)
    sorted_idx = np.argsort(p) # argsort gives indices
    
    max_obj = -np.inf
    best_b = np.zeros(K)
    
    for k in range(K, K + 1):
        if Btot < k:
            active_count = Btot
        else:
            active_count = k
            
        current_b = np.zeros(K)
        base_bits = Btot // active_count
        rem_bits = Btot % active_count
        
        active_users = sorted_idx[:active_count]
        
        for i in range(active_count):
            u_idx = active_users[i]
            extra = 1 if i < rem_bits else 0
            current_b[u_idx] = base_bits + extra
            
        current_obj = objective_bit_allocation(current_b, p, D)
        
        if current_obj > max_obj:
            max_obj = current_obj
            best_b = current_b.copy()
            
    return max_obj, best_b

# 5) Top-K-LAG (Waterfilling + Greedy Rounding + Local Search)
def solve_1d_for_user(i, lam, p, D, Bmax):
    # Golden Section Search to maximize: f_i(b) - lambda * b
    if Bmax <= 0:
        return 0.0
    
    p_i = p[i]
    def obj(x):
        return f_fun_safe(x, p_i, D) - lam * x
    
    a, b_bound = 0.0, float(Bmax)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    
    x1 = a + resphi * (b_bound - a)
    x2 = b_bound - resphi * (b_bound - a)
    f1 = obj(x1)
    f2 = obj(x2)
    
    tol = 1e-4
    max_iter = 50
    
    for _ in range(max_iter):
        if abs(b_bound - a) < tol:
            break
        if f1 < f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b_bound - resphi * (b_bound - a)
            f2 = obj(x2)
        else:
            b_bound = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b_bound - a)
            f1 = obj(x1)
            
    # Check candidates
    candidates = [a, b_bound, x1, x2]
    cand_vals = [obj(c) for c in candidates]
    idx_max = np.argmax(cand_vals)
    b_star = candidates[idx_max]
    
    # Compare with b=0
    f0 = obj(0.0)
    f_star = obj(b_star)
    
    if f0 >= f_star:
        return 0.0
    else:
        return max(0.0, b_star)

def waterfill_on_set(Sk, Btot, p, D):
    K = len(p)
    b_cont = np.zeros(K)
    
    lambda_low = 0.0
    lambda_high = 1e6
    
    # Bisection search for lambda
    for _ in range(50):
        lam = (lambda_low + lambda_high) / 2.0
        sum_b = 0.0
        
        # For each active user, solve 1D problem
        for u_idx in Sk:
            b_i = solve_1d_for_user(u_idx, lam, p, D, Btot)
            b_cont[u_idx] = b_i
            sum_b += b_i
            
        if sum_b > Btot:
            lambda_low = lam
        else:
            lambda_high = lam
            
    # Scale to match sum exactly (if not zero)
    sum_b = np.sum(b_cont[Sk])
    if sum_b > 0:
        scale = Btot / sum_b
        b_cont[Sk] *= scale
        
    return b_cont

def round_bits_to_integer_greedy(b_cont, Btot, Sk, p, D):
    # Greedy Rounding using Marginal Gain
    b_int = np.floor(b_cont).astype(int)
    current_sum = np.sum(b_int)
    rem_bits = int(Btot - current_sum)
    
    if rem_bits <= 0:
        return b_int
        
    for _ in range(rem_bits):
        best_gain = -np.inf
        best_u = -1
        
        for u in Sk:
            val_curr = f_fun_safe(b_int[u], p[u], D)
            val_next = f_fun_safe(b_int[u] + 1, p[u], D)
            gain = val_next - val_curr
            
            if gain > best_gain:
                best_gain = gain
                best_u = u
                
        if best_u != -1:
            b_int[best_u] += 1
            
    return b_int

def local_search_1bit(b_init, Sk, p, D, Btot):
    b_ls = b_init.copy()
    K = len(p)
    f_vals = np.zeros(K)
    for i in range(K):
        f_vals[i] = f_fun(b_ls[i], p[i], D)
    F_current = np.sum(f_vals)
    
    max_iter = Btot * len(Sk)
    
    for _ in range(max_iter):
        best_delta = 0.0
        best_i, best_j = -1, -1
        best_fi_after, best_fj_after = 0.0, 0.0
        
        # Try moving 1 bit from i to j (i, j in Sk)
        for i in Sk:
            if b_ls[i] <= 0: continue
            
            fi_before = f_vals[i]
            fi_after = f_fun(b_ls[i] - 1, p[i], D)
            delta_i = fi_after - fi_before
            
            for j in Sk:
                if i == j: continue
                
                fj_before = f_vals[j]
                fj_after = f_fun(b_ls[j] + 1, p[j], D)
                
                delta_total = delta_i + (fj_after - fj_before)
                
                if delta_total > best_delta + 1e-12:
                    best_delta = delta_total
                    best_i = i
                    best_j = j
                    best_fi_after = fi_after
                    best_fj_after = fj_after
                    
        if best_delta <= 0:
            break
            
        # Apply move
        b_ls[best_i] -= 1
        b_ls[best_j] += 1
        f_vals[best_i] = best_fi_after
        f_vals[best_j] = best_fj_after
        F_current += best_delta
        
    return b_ls, F_current

def solve_topk_waterfilling(Btot, p, D):
    K = len(p)
    sorted_idx = np.argsort(p)
    
    F_best = -np.inf
    b_best = np.zeros(K)
    
    K_max = min(K, Btot)
    
    for k in range(1, K_max + 1):
        Sk = sorted_idx[:k] # Top-k user indices
        
        # Strategy 1: Lagrangian Waterfilling + Greedy Rounding
        b_cont = waterfill_on_set(Sk, Btot, p, D)
        b_lag = round_bits_to_integer_greedy(b_cont, Btot, Sk, p, D)
        
        F_lag = objective_bit_allocation(b_lag, p, D)
        
        # Strategy 2: Uniform Distribution
        b_eq = np.zeros(K)
        active_count = min(k, Btot)
        base_bits = Btot // active_count
        rem_bits = Btot % active_count
        
        # Sk is already sorted by quality, so just take first active_count
        # Note: Sk has length k. 
        active_users_eq = Sk[:active_count]
        
        for idx, u in enumerate(active_users_eq):
            extra = 1 if idx < rem_bits else 0
            b_eq[u] = base_bits + extra
            
        F_eq = objective_bit_allocation(b_eq, p, D)
        
        # Hybrid Selection
        if F_eq >= F_lag:
            F_curr = F_eq
            b_k = b_eq
        else:
            F_curr = F_lag
            b_k = b_lag
            
        # Local Search Improvement
        b_k, F_ls = local_search_1bit(b_k, Sk, p, D, Btot)
        F_curr = F_ls
        
        # Global Update
        if F_curr > F_best:
            F_best = F_curr
            b_best = b_k.copy()
            
    return F_best, b_best
