import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# --- 1. Disturbance Sampling Function ---
def sample_zonotope(c_w, G_w):
    """
    Sample w = c_w + G_w * xi, where xi ~ Unif([-1,1]^p).
    c_w: (n,) or (n,1)
    G_w: (n,p)
    returns: (n,)
    """
    c_w = np.asarray(c_w).reshape(-1)
    G_w = np.asarray(G_w)
    # The number of generators (p) is the number of columns in G_w
    p = G_w.shape[1] 
    xi = np.random.uniform(-1.0, 1.0, size=(p,))
    return c_w + G_w @ xi


# Define the polyhedral matrix S for n=3 (State Constraint Set)
S = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
])
b_S = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]
])

# Dimensions
n, m = 3, 3  # State dimension (n=3), Input dimension (m=3)
q = S.shape[0] 

# Known matrices (The true system, used only for data generation)
A_true = np.array([[0.8, 0.1, 0.0],
                   [-0.2, 0.9, 0.1],
                   [0.1, 0.0, 0.7]]) 
B_true = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]) 

# --- 2. Disturbance Zonotope Definition (used for sampling) ---
c_w = np.zeros((n, 1)) # Center is zero
G_w = 0.01 * np.eye(n) # Generator is diagonal with small bounds (n x n)
# The dimension of the disturbance generator is p=n (3)

# y = np.abs(S @ G_w) 
SW = S @ c_w
M_x = 6

# --- 3. Modified Data Generation Function ---
def generate_linear_data(A, B, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5):
    n, m = B.shape

    x = np.zeros((num_steps + 1, n))
    u = np.zeros((num_steps, m))
    w = np.zeros((num_steps, n))

    x[0] = np.asarray(x0).reshape(-1)

    for k in range(num_steps):
        # Sample w from the Zonotope defined by c_w and G_w
        w[k] = sample_zonotope(c_w, G_w)
        u[k] = np.random.uniform(u_min, u_max, size=(m,))
        x[k + 1] = A @ x[k] + B @ u[k] + w[k]

    # Build data matrices AFTER the loop
    N = num_steps
    X_k1 = x[1:N + 1].T   # x_{k+1} (n x N)
    X_k  = x[0:N].T       # x_k     (n x N)
    U_k  = u[0:N].T       # u_k     (m x N)
    return X_k, X_k1, U_k

# Generate linear data
num_steps = 10
x0 = 0.5 * np.array([1.0, 1.0, 1.0]) 

# Data is generated using the 'true' system and the new zonotopic disturbance
X_k, X_k1, U_k = generate_linear_data(A_true, B_true, x0, num_steps, c_w, G_w)

# Nominal Zonotope Parameters (Your Guess)
Delta_A = 0.15 
Delta_B = 0.10 
A_nom = np.array([[0.9, 0.0, 0.0],
                  [-0.1, 0.9, 0.05],
                  [0.05, 0.0, 0.75]])     
B_nom = np.array([[0.9, 0.1, 0.0],
                  [0.0, 1.1, 0.0],
                  [0.0, 0.0, 0.9]])     

c_A = A_nom    
G_A = Delta_A * np.eye(n) 
c_B = B_nom    
G_B = Delta_B * np.eye(n) 

n, N = X_k.shape

def COP_Side_AB_lambda_Zonotopes(X_k1, X_k, U_k, n, m, c_A, c_B, G_A, G_B):
    # --- State Constraint Set Calculation (Full Data Convex Hull) ---
    if X_k1.shape[0] == 1:
        # 1D case 
        x_all = np.concatenate([X_k.reshape(-1), X_k1.reshape(-1)])
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size == 0:
            raise ValueError("No finite data points found in X_k/X_k1.")

        h_min = float(np.min(x_all))
        h_max = float(np.max(x_all))

        scale = max(1.0, abs(h_min), abs(h_max), float(np.max(np.abs(x_all))))
        eps = 1e-12 * scale 

        H_X = np.array([[1.0], [-1.0]])
        h_X = np.array([[h_max + eps],
                        [-(h_min - eps)]])
    else:
        # General multi-D case (n=3)
        X_all = np.hstack((X_k, X_k1)) 
        hull_X = ConvexHull(X_all.T) 
        H_X = hull_X.equations[:, :-1] 
        h_X = -hull_X.equations[:, [-1]] 
    # --- END State Constraint Set Calculation ---

    eta_w = n
    G_fixed = np.eye(n) 
    ones_vector = np.ones((1, eta_w))
    
    # Decision variables
    c = cp.Variable((n, 1))
    c1 = cp.Variable((n, n)) 
    c2 = cp.Variable((n, m)) 
    s_X = cp.Variable(nonneg=True)

    N_data = X_k.shape[1] 
    λ = cp.Variable((eta_w, N_data)) 
    λ1 = cp.Variable((N_data, n))    
    λ2 = cp.Variable((N_data, m))    
    λ_max = cp.Variable((n, 1))      
    λ_max1 = cp.Variable((1, n))     
    λ_max2 = cp.Variable((1, m))     

    constraints = []

    # Maximal generators 
    G_A1 = np.eye(n)    
    G_B1 = np.ones((n, m)) 

    # Lambda max constraints
    constraints += [λ_max >= cp.abs(λ)]
    constraints += [λ_max1 >= cp.abs(λ1)]
    constraints += [λ_max2 >= cp.abs(λ2)]

    # Zonotope containment constraint
    constraints += [
        H_X @ c + H_X @ G_fixed @ λ_max <= s_X * h_X 
    ]

    # Dynamics constraints for each timestep
    for k in range(N_data):
        
        Gamma_A = cp.Variable((G_A.shape[1], G_A.shape[1])) 
        gamma_A = cp.Variable((G_A.shape[1], 1))             
        Gamma_B = cp.Variable((G_B.shape[1], G_B.shape[1])) 
        gamma_B = cp.Variable((G_B.shape[1], 1))             

        A_zono_k = c1 + λ1[[k], :] @ G_A1
        B_zono_k = c2 + G_B1 @ λ2[[k], :].T 

        # The core constraints: The estimated zonotope is contained within the Nominal Zonotope
        constraints += [
            # Containment for A
            λ1[[k], :] @ G_A1 == G_A @ Gamma_A, 
            c_A - c1 == G_A @ gamma_A,           
            cp.norm(cp.hstack([Gamma_A, gamma_A]), p='inf') <= 1, 
            
            # Containment for B
            G_B1 @ λ2[[k], :].T == G_B @ Gamma_B, 
            c_B - c2 == G_B @ gamma_B,             
            cp.norm(cp.hstack([Gamma_B, gamma_B]), p='inf') <= 1 
        ]

        # Prediction error constraint
        constraints += [
            X_k1[:, [k]] - A_zono_k @ X_k[:, [k]] - B_zono_k @ U_k[:, [k]] == c + G_fixed @ λ[:, [k]],
            λ[:, [k]] <= 1,
            λ[:, [k]] >= -1,
            λ1[[k], :] <= 1,
            λ1[[k], :] >= -1,
            λ2[[k], :] <= 1,
            λ2[[k], :] >= -1
        ]

    # Cost function
    J_M = 0.5 * ones_vector @ λ_max +  0.5 * s_X 
    + 0.5 * ones_vector @ λ_max1.T + 0.5 * ones_vector @ λ_max2.T

    # Optimization problem
    problem = cp.Problem(cp.Minimize(J_M), constraints)

    # Solve
    try:
        problem.solve(solver=cp.MOSEK, mosek_params={"MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8})
    except Exception as e:
        print(f"MOSEK solver failed, attempting ECOS. Error: {e}")
        problem.solve(solver=cp.ECOS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Solver status: {problem.status}")
        return None, None, None, None, None, None, None, None, None, None, None

    return c1.value, c2.value, c.value, s_X.value, λ_max.value, λ_max1.value, λ_max2.value, gamma_A.value, Gamma_A.value, gamma_B.value, Gamma_B.value


# --- Execute the modified optimization ---
c1_side, c2_side, c_side, s_X, λ_max_side, λ_max1_side, λ_max2_side, gamma_A, Gamma_A, gamma_B, Gamma_B = COP_Side_AB_lambda_Zonotopes(X_k1, X_k, U_k, n, m, c_A, c_B, G_A, G_B)

# # Print results if successful
# if c1 is not None:
#     print("✅ Optimization Successful! Results for the 3x3 system using Zonotopic Disturbance:")
#     print("--- Optimal Estimated Center A (c1) ---")
#     print(c1)
#     print("--- Optimal Estimated Center B (c2) ---")
#     print(c2)
#     print("\n--- Optimal Residual Center (c) ---")
#     print(c)
#     print(f"\n--- Optimal Scaling Factor (s_X) ---")
#     print(s_X)
#     print("\n--- Max Residual Lambda (λ_max) ---")
#     print(λ_max)
# else:
#     print("❌ Optimization failed.")


def COP_lambda_Zonotopes(X_k1, X_k, U_k, n, m):
    # --- State Constraint Set Calculation (Full Data Convex Hull) ---
    if X_k1.shape[0] == 1:
        # 1D case 
        x_all = np.concatenate([X_k.reshape(-1), X_k1.reshape(-1)])
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size == 0:
            raise ValueError("No finite data points found in X_k/X_k1.")

        h_min = float(np.min(x_all))
        h_max = float(np.max(x_all))

        scale = max(1.0, abs(h_min), abs(h_max), float(np.max(np.abs(x_all))))
        eps = 1e-12 * scale 

        H_X = np.array([[1.0], [-1.0]])
        h_X = np.array([[h_max + eps],
                        [-(h_min - eps)]])
    else:
        # General multi-D case (n=3)
        X_all = np.hstack((X_k, X_k1)) 
        hull_X = ConvexHull(X_all.T) 
        H_X = hull_X.equations[:, :-1] 
        h_X = -hull_X.equations[:, [-1]] 
    # --- END State Constraint Set Calculation ---

    eta_w = n
    G_fixed = np.eye(n) 
    ones_vector = np.ones((1, eta_w))
    
    # Decision variables
    c = cp.Variable((n, 1))
    s_X = cp.Variable(nonneg=True)
    A = cp.Variable((n, n))  # System matrix A
    B = cp.Variable((n, m))  # Control matrix B

    N_data = X_k.shape[1] 
    λ = cp.Variable((eta_w, N_data))    
    λ_max = cp.Variable((n, 1))     

    constraints = []

    # Lambda max constraints
    constraints += [λ_max >= cp.abs(λ)]

    # Zonotope containment constraint
    constraints += [
        H_X @ c + H_X @ G_fixed @ λ_max <= s_X * h_X 
    ]

    # Dynamics constraints for each timestep
    for k in range(N - 1):

        constraints += [
            X_k1[:, [k]] - A @ X_k[:, [k]] - B @ U_k[:, [k]] == c + G_fixed @ λ[:, [k]],
            λ[:, [k]] <= 1,
            λ[:, [k]] >= -1,
        ]

    # Cost function
    J_M = 0.5 * ones_vector @ λ_max +  0.5 * s_X 

    # Optimization problem
    problem = cp.Problem(cp.Minimize(J_M), constraints)

    # Solve
    try:
        problem.solve(solver=cp.MOSEK, mosek_params={"MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8})
    except Exception as e:
        print(f"MOSEK solver failed, attempting ECOS. Error: {e}")
        problem.solve(solver=cp.ECOS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Solver status: {problem.status}")
        return None, None, None, None, None, None

    return A.value, B.value, c.value, s_X.value, λ_max.value, λ.value


A1, B1, c, s_X, λ_max, λ = COP_lambda_Zonotopes(X_k1, X_k, U_k, n, m)
# print("Optimal A:\n", A1)
# print("Optimal B:\n", B1)
# print("Optimal λ_max:\n", λ_max)
# print("Optimal c:\n", c)


# exit()


# Parameter sweep
lambda_values = np.linspace(0.7, 1.0, num=10)
num_trials = 100

baseline_feas = []
extended_feas = []

for lamda in lambda_values:
    count_baseline = 0
    count_extended = 0

    for trial in range(num_trials):
        G_wnew = G_w + np.random.uniform(-0.05, 0.05, size=G_w.shape)

        # Compute y for current G_wnew
        y = np.zeros((q, m))
        for j in range(q):
            for i in range(n):
                y[j, 0] += S[j, :] @ G_wnew[:, i]
        y = np.abs(y)


        # === Baseline method ===
        K = cp.Variable((m, n))
        P = cp.Variable((q, q), nonneg=True)
        rho = cp.Variable(nonneg=True)

        lhs = P @ b_S
        rhs = lamda * b_S - SW - y

        constraints = [
            lhs <= rhs,
            P @ S == S @ (A1 + B1 @ K),
            cp.norm(K, 'fro') <= rho
        ]

        problem = cp.Problem(cp.Minimize(rho), constraints)
        try:
            problem.solve(solver=cp.MOSEK)
            if problem.status == cp.OPTIMAL:
                count_baseline += 1
        except:
            pass

        # === Extended method ===
        K2 = cp.Variable((m, n))
        P2 = cp.Variable((q, q), nonneg=True)
        rho2 = cp.Variable(nonneg=True)

        c_kclosed = c1_side + c2_side @ K2
        G_A = np.diag(λ_max1_side.flatten())
        G_B = np.diag(λ_max2_side.flatten())

        lhs2 = P2 @ b_S
        rhs2 = (
            lamda * b_S - SW - y
            - M_x * (S @ G_A[:, [0]] + S @ G_A[:, [1]])
            - rho2 * M_x * (S @ G_B)
        )

        constraints2 = [
            lhs2 <= rhs2,
            P2 @ S == S @ c_kclosed,
            cp.norm(K2, 'fro') <= rho2
        ]

        problem2 = cp.Problem(cp.Minimize(rho2), constraints2)
        try:
            problem2.solve(solver=cp.MOSEK)
            if problem2.status == cp.OPTIMAL:
                count_extended += 1
        except:
            pass

    baseline_feas.append(100 * count_baseline / num_trials)
    extended_feas.append(100 * count_extended / num_trials)

    print(f"α = {lamda:.2f} → Baseline: {baseline_feas[-1]:.1f}% | Extended: {extended_feas[-1]:.1f}%")

# === Plotting ===
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, baseline_feas, marker='o', label='Conformant model')
plt.plot(lambda_values, extended_feas, marker='s', label='Data and Side information Conformant model')
plt.xlabel(r'$\lambda$ ')
plt.ylabel('Feasibility (%)')
plt.grid(True)
plt.xlim(0.7, 1)
plt.ylim(0, 100)
plt.legend()
plt.show()


exit()



# Parameter sweep
alpha_values = np.linspace(1.0, 5.0, 9)
num_trials = 100

baseline_feas = []
extended_feas = []

for alpha in alpha_values:
    count_baseline = 0
    count_extended = 0

    for trial in range(num_trials):
        G_wnew = alpha * G_w + np.random.uniform(-0.05, 0.05, size=G_w.shape)

        # Compute y for current G_wnew
        y = np.zeros((q, m))
        for j in range(q):
            for i in range(n):
                y[j, 0] += S[j, :] @ G_wnew[:, i]
        y = np.abs(y)

        lamda = 0.8

        # === Baseline method ===
        K = cp.Variable((m, n))
        P = cp.Variable((q, q), nonneg=True)
        rho = cp.Variable(nonneg=True)

        lhs = P @ b_S
        rhs = lamda * b_S - SW - y

        constraints = [
            lhs <= rhs,
            P @ S == S @ (A1 + B1 @ K),
            cp.norm(K, 'fro') <= rho
        ]

        problem = cp.Problem(cp.Minimize(rho), constraints)
        try:
            problem.solve(solver=cp.MOSEK)
            if problem.status == cp.OPTIMAL:
                count_baseline += 1
        except:
            pass

        # === Extended method ===
        K2 = cp.Variable((m, n))
        P2 = cp.Variable((q, q), nonneg=True)
        rho2 = cp.Variable(nonneg=True)

        c_kclosed = c1_side + c2_side @ K2
        G_A = np.diag(λ_max1_side.flatten())
        G_B = np.diag(λ_max2_side.flatten())

        lhs2 = P2 @ b_S
        rhs2 = (
            lamda * b_S - SW - y
            - M_x * (S @ G_A[:, [0]] + S @ G_A[:, [1]])
            - rho2 * M_x * (S @ G_B)
        )

        constraints2 = [
            lhs2 <= rhs2,
            P2 @ S == S @ c_kclosed,
            cp.norm(K2, 'fro') <= rho2
        ]

        problem2 = cp.Problem(cp.Minimize(rho2), constraints2)
        try:
            problem2.solve(solver=cp.MOSEK)
            if problem2.status == cp.OPTIMAL:
                count_extended += 1
        except:
            pass

    baseline_feas.append(100 * count_baseline / num_trials)
    extended_feas.append(100 * count_extended / num_trials)

    print(f"α = {alpha:.2f} → Baseline: {baseline_feas[-1]:.1f}% | Extended: {extended_feas[-1]:.1f}%")

# === Plotting ===
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, baseline_feas, marker='o', label='Conformant model')
plt.plot(alpha_values, extended_feas, marker='s', label='Data and Side information Conformant model')
plt.xlabel(r'$\alpha_1$ (Noise scaling factor)')
plt.ylabel('Feasibility (%)')
plt.grid(True)
plt.xlim(1, 5)
plt.ylim(0, 100)
plt.legend()
plt.show()