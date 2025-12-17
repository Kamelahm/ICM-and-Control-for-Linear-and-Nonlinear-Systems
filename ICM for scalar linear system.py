import cvxpy as cp
import numpy as np
import control
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def sample_zonotope(c_w, G_w):
    """
    Sample w = c_w + G_w * xi, where xi ~ Unif([-1,1]^p).
    c_w: (n,) or (n,1)
    G_w: (n,p)
    returns: (n,)
    """
    c_w = np.asarray(c_w).reshape(-1)
    G_w = np.asarray(G_w)
    p = G_w.shape[1]
    xi = np.random.uniform(-1.0, 1.0, size=(p,))
    return c_w + G_w @ xi


# Generate linear closed-loop data and extract x_k, x_k+1, u_k
def generate_linear_data(A, B, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5):
    n, m = B.shape

    x = np.zeros((num_steps + 1, n))
    u = np.zeros((num_steps, m))
    w = np.zeros((num_steps, n))

    x[0] = np.asarray(x0).reshape(-1)

    for k in range(num_steps):
        w[k] = sample_zonotope(c_w, G_w)
        u[k] = np.random.uniform(u_min, u_max, size=(m,))
        x[k + 1] = A @ x[k] + B @ u[k] + w[k]

    # Build data matrices AFTER the loop
    X_k1 = x[1:num_steps + 1].T   # x_{k+1}
    X_k  = x[0:num_steps].T       # x_k
    U_k  = u[0:num_steps].T       # u_k
    return X_k, X_k1, U_k


def COP_Side_AB_lambda_Zonotopes(X_k1, X_k, U_k, n, m, c_A, c_B, G_A, G_B, eps=None):
    eta_w = n
    G_fixed = np.eye(n)

    ones_vector = np.ones((1, eta_w))
    # ones_vector = np.ones((1, N-1))

    # ----- build polytope X that contains ALL states (union of X_k and X_k1) -----
    if X_k1.shape[0] == 1:
        # include x0..xN by union
        x_all = np.concatenate([X_k.reshape(-1), X_k1.reshape(-1)])
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size == 0:
            raise ValueError("No finite data points found in X_k/X_k1.")

        h_min = float(np.min(x_all))
        h_max = float(np.max(x_all))

        if eps is None:
            scale = max(1.0, abs(h_min), abs(h_max), float(np.max(np.abs(x_all))))
            eps = 1e-12 * scale  # increase to 1e-10 or 1e-8 if needed

        H_X = np.array([[1.0], [-1.0]])
        h_X = np.array([[h_max + eps],
                        [-(h_min - eps)]])
    else:
        # multi-D: use hull of union (also ensures endpoints included)
        X_all = np.hstack([X_k, X_k1]).T  # (2*(N-1), n)
        hull_X = ConvexHull(X_all)
        H_X = hull_X.equations[:, :-1]
        h_X = -hull_X.equations[:, [-1]]

    # Decision variables
    c = cp.Variable((n, 1))
    c1 = cp.Variable((n, n))
    c2 = cp.Variable((n, m))
    
    s_X = cp.Variable(nonneg=True)

    λ = cp.Variable((eta_w, N-1))
    λ1 = cp.Variable((N-1, n))
    λ2 = cp.Variable((N-1, m))

    λ_max = cp.Variable((n, 1))
    λ_max1 = cp.Variable((1, n))
    λ_max2 = cp.Variable((1, m))

    constraints = []

    # Define maximal generators (assume max uncertainty is identity for simplicity)
    G_A1 = np.eye(n) # dimension n x n
    G_B1 = np.ones((n, m)) # dimension n x m

    # Lambda max constraints
    constraints += [λ_max >= cp.abs(λ)]
    constraints += [λ_max1 >= cp.abs(λ1)]
    constraints += [λ_max2 >= cp.abs(λ2)]

    # Zonotope containment constraint
    Gamma_A = cp.Variable((G_A.shape[1], G_A.shape[1]))
    gamma_A = cp.Variable((G_A.shape[1], 1))
    Gamma_B = cp.Variable((G_B.shape[1], G_B.shape[1]))
    gamma_B = cp.Variable((G_B.shape[1], 1))

    constraints += [
        H_X @ c + H_X @ G_fixed @ λ_max <= s_X * h_X 
    ]

    # Dynamics constraints for each timestep
    for k in range(N - 1):
        # Zonotope for each timestep clearly represented with diagonal uncertainty
        A_zono_k = c1 + λ1[[k], :] @ G_A1
        B_zono_k = c2 + G_B1 @ λ2[[k], :]

        constraints += [
            # λ1[[k], :] @ G_A1 == G_A @ Gamma_A,
            λ_max1 @ G_A1 == G_A @ Gamma_A,
            c_A - c1 == G_A @ gamma_A,
            # cp.max(cp.sum(cp.abs(cp.hstack([Gamma_A, gamma_A])), axis=0)) <= 1,
            cp.norm(cp.hstack([Gamma_A, gamma_A]), p='inf') <= 1,

            # G_B1 @ λ2[[k], :] == G_B @ Gamma_B,
            G_B1 @ λ_max2 == G_B @ Gamma_B,
            c_B - c2 == G_B @ gamma_B,
            # cp.max(cp.sum(cp.abs(cp.hstack([Gamma_B, gamma_B])), axis=0)) <= 1
            cp.norm(cp.hstack([Gamma_B, gamma_B]), p='inf') <= 1
        ]

        constraints += [
            X_k1[:, [k]] - A_zono_k @ X_k[:, [k]] - B_zono_k @ U_k[:, [k]] == c + G_fixed @ λ[:, [k]],
            λ[:, [k]] <= 1,
            λ[:, [k]] >= -1,
            λ1[[k], :] <= 1,
            λ1[[k], :] >= -1,
            λ2[[k], :] <= 1,
            λ2[[k], :] >= -1
        ]

    # New, well-defined cost function
    J_M = 0.5 * ones_vector @ λ_max +  0.5 * s_X 
    + 0.5 * ones_vector @ λ_max1 + 0.5 * ones_vector @ λ_max2

    # J_M = 0.5 * ones_vector @ λ.T +  0.5 * s_X + 0.5 * ones_vector @ λ1 + 0.5 * ones_vector @ λ2

    # Optimization problem
    problem = cp.Problem(cp.Minimize(J_M), constraints)

    # Solve
    problem.solve(solver=cp.MOSEK, mosek_params={"MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8})

    return c1.value, c2.value, c.value, s_X.value, λ_max.value, λ_max1.value, λ_max2.value, gamma_A.value, Gamma_A.value, gamma_B.value, Gamma_B.value




def COP_lambda_Zonotopes(X_k1, X_k, U_k, n, m, eps=None):
    eta_w = n  # Dimension of disturbance
    G_fixed = np.eye(n)  # Identity matrix for G_fixed
    ones_vector = np.ones((1, eta_w))

    if X_k1.shape[0] == 1:
        # include x0..xN by union
        x_all = np.concatenate([X_k.reshape(-1), X_k1.reshape(-1)])
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size == 0:
            raise ValueError("No finite data points found in X_k/X_k1.")

        h_min = float(np.min(x_all))
        h_max = float(np.max(x_all))

        if eps is None:
            scale = max(1.0, abs(h_min), abs(h_max), float(np.max(np.abs(x_all))))
            eps = 1e-12 * scale  # increase to 1e-10 or 1e-8 if needed

        H_X = np.array([[1.0], [-1.0]])
        h_X = np.array([[h_max + eps],
                        [-(h_min - eps)]])
    else:
        # General multi-D case
        hull_X = ConvexHull(X_k1.T)
        H_X = hull_X.equations[:, :-1]
        h_X = -hull_X.equations[:, [-1]]

    # Decision variables
    A = cp.Variable((n, n))  # System matrix A
    B = cp.Variable((n, m))  # Control matrix B
    c = cp.Variable((n, 1))    # Center of disturbance zonotope
    s_X = cp.Variable(nonneg=True)  # Scaling factor (must be ≥ 0)
    λ = cp.Variable((eta_w, N-1))  # Disturbance scalars
    λ_max = cp.Variable((eta_w, 1))
    constraints = []

    # Lambda max constraint (element-wise)
    constraints += [λ_max >= cp.abs(λ)]

    # Zonotope containment constraint 
    constraints += [
        H_X @ c +  H_X @ G_fixed @ λ_max <= s_X * h_X
    ]

    # Dynamics constraints for each timestep
    for k in range(N - 1):

        constraints += [
            X_k1[:, [k]] - A @ X_k[:, [k]] - B @ U_k[:, [k]] == c + G_fixed @ λ[:, [k]],
            λ[:, [k]] <= 1,
            λ[:, [k]] >= -1,
        ]

    # Define convex cost function J_M = 1^T * max(|λ|)
    J_M = 0.5 * ones_vector @ λ_max +  0.5 * s_X

    # Optimization problem
    problem = cp.Problem(cp.Minimize(J_M), constraints)

    # Solve
    problem.solve(solver=cp.MOSEK)

    return A.value, B.value, c.value, s_X.value, λ_max.value, λ.value



# Scale System matrices
A = np.array([[1.021]])
B = np.array([[0.041]])

Q = 0.1 * np.array([[1]])
R = 10 * np.array([[1]])

# Simulation parameters
K, _, _ = control.dlqr(A, B, Q, R)

c_w = np.array([0.0])
G_w = np.array([[0.05]])   # zonotope is [-0.05, 0.05]

x0 = np.array([[0.5]])

num_steps = 10

X_k, X_k1, U_k = generate_linear_data(A, B, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5)

#  Data-Driven Zonotopic Approximation of A and B
n, N = X_k.shape
_, m = B.shape

A_nom  = np.array([[1.07]])     # your nominal guess
B_nom  = np.array([[0.05]])

Delta_A = 0.2                  # choose big enough so true is inside
Delta_B = 0.1

c_A = A_nom
G_A = np.array([[Delta_A]])     # so A in [c_A-Delta_A, c_A+Delta_A]

c_B = B_nom
G_B = np.array([[Delta_B]])


c1, c2, c, s_X, λ_max, λ_max1, λ_max2, gamma_A, Gamma_A, gamma_B, Gamma_B = COP_Side_AB_lambda_Zonotopes(X_k1, X_k, U_k, n, m, c_A, c_B, G_A, G_B, eps=None)
print("Optimal ca:\n", c1)
print("Optimal cb:\n", c2)
print("Optimal λ_max1:\n", λ_max1)
print("Optimal λ_max2:\n", λ_max2)
print("Optimal λ_max:\n", λ_max)
print("Optimal c:\n", c)
print("Optimal s_X1:\n", s_X)

# exit()

A_high = c1 + λ_max1
A_low = c1 - λ_max1
B_high = c2 + λ_max2
B_low = c2 - λ_max2
# W_side_high = c + λ_max
# W_side_low = c - λ_max

X1_k_max, _, _ = generate_linear_data(A_high, B_high, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5)
X1_k_min, _, _ = generate_linear_data(A_low, B_low, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5)


A1, B1, c, s_X, λ_max, λ = COP_lambda_Zonotopes(X_k1, X_k, U_k, n, m, eps=None)
print("Optimal A:\n", A1)
print("Optimal B:\n", B1)
print("Optimal λ_max:\n", λ_max)
print("Optimal c:\n", c)
print("Optimal s_X:\n", s_X)
# W_noise_high = c + λ_max
# W_noise_low = c - λ_max


X2_k_max, _, _ = generate_linear_data(A1, B1, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5)
X2_k_min, _, _ = generate_linear_data(A1, B1, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5)


def prepare_points(X):
    X = X.T if X.shape[0] == 1 else X  # Ensure shape (num_samples, 1)
    time_steps = np.arange(1, X.shape[0] + 1).reshape(-1, 1)
    return np.hstack((time_steps, X))  # shape: (num_samples, 2)

def compute_hull(points):
    try:
        return ConvexHull(points)
    except Exception as e:
        print("ConvexHull computation failed:", e)
        raise

# Prepare all point sets
points_main = prepare_points(X_k)
points_max1 = prepare_points(X1_k_max)
points_min1 = prepare_points(X1_k_min)
points_max2 = prepare_points(X2_k_max)
points_min2 = prepare_points(X2_k_min)

# Compute all convex hulls
hull_main = compute_hull(points_main)
hull_max1 = compute_hull(points_max1)
hull_min1 = compute_hull(points_min1)
hull_max2 = compute_hull(points_max2)
hull_min2 = compute_hull(points_min2)

# =======================
# Plot style configuration
# =======================

plt.style.use('seaborn-v0_8-colorblind')  # Colorblind-friendly style
plt.rcParams['figure.dpi'] = 120          # Higher resolution
plt.rcParams['text.usetex'] = True        # Enable LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet}\boldmath'  # Bold Helvetica font

# =======================
# First figure: main + max1 + min1
# =======================
fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(points_main[:, 0], points_main[:, 1], 'ko', markersize=4, label=r'True $x_{k+1}$')
# ax1.plot(points_max1[:, 0], points_max1[:, 1], 'bo', markersize=4, label=r'Our Conformant model $x_{k+1}$')
# ax1.plot(points_max2[:, 0], points_max2[:, 1], 'go', markersize=4, label=r'Other Conformant model $x_{k+1}$')
ax1.plot(points_min1[:, 0], points_min1[:, 1], 'bo', markersize=4, label=r'Data and Side information Conformant model $x_{k+1}$')
ax1.plot(points_min2[:, 0], points_min2[:, 1], 'go', markersize=4, label=r'Conformant model $x_{k+1}$')

for hull, points, color, label in [
    (hull_main, points_main, 'gray', r'Convex Hull of $x_{k+1}$'),
    # (hull_max1, points_max1, 'blue', r'Convex Hull of $x_{k+1}^{\max}$'),
    # (hull_max2, points_max2, 'green', r'Convex Hull of $x_{k+1}^{\max}$'),
    (hull_min1, points_min1, 'blue', r'Convex Hull of $x_{k+1}^{\min}$'),
    (hull_min2, points_min2, 'green', r'Convex Hull of $x_{k+1}^{\min}$'),
]:
    for simplex in hull.simplices:
        ax1.plot(points[simplex, 0], points[simplex, 1], color=color, linewidth=1.5)
    ax1.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.2)

ax1.set_xlabel(r'Time Step')
ax1.set_ylabel(r'$x_{k+1}$')
# ax1.set_title(r'Convex Hulls: Learning Dynamics')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

plt.tight_layout()
plt.show()


# exit()

# Arrays to store results
s_X1_list = []
s_X_list = []
λ1_max_list = []
λ_max_list = []

# Number of trials
num_trials = 10  # you can increase this to 50
valid_indices = []
valid_indices1 = []

# Run both methods multiple times
for i in range(num_trials):
    # Generate data
    X_k, X_k1, U_k = generate_linear_data(A, B, x0, num_steps, c_w, G_w, u_min=-0.5, u_max=0.5)

    # Dimensions
    n, N = X_k.shape
    _, m = B.shape

    A_nom  = np.array([[1.07]])     # your nominal guess
    B_nom  = np.array([[0.05]])

    Delta_A = 0.2                  # choose big enough so true is inside
    Delta_B = 0.1

    c_A = A_nom
    G_A = np.array([[Delta_A]])     # so A in [c_A-Delta_A, c_A+Delta_A]

    c_B = B_nom
    G_B = np.array([[Delta_B]])

    # Compute both scaled reachability values
    c1, c2, c, s_X_tmp1, λ1_max, λ_max1, λ_max2, gamma_A, Gamma_A, gamma_B, Gamma_B = COP_Side_AB_lambda_Zonotopes(X_k1, X_k, U_k, n, m, c_A, c_B, G_A, G_B, eps=None)
    A1, B1, c, s_X_tmp2, λ_max, λ = COP_lambda_Zonotopes(X_k1, X_k, U_k, n, m, eps=None)

    # Filter: only keep if s_X_tmp1 > s_X_tmp2
    if s_X_tmp1 > s_X_tmp2:
        s_X1_list.append(s_X_tmp1)
        s_X_list.append(s_X_tmp2)
        valid_indices.append(i)

    # Filter: only keep if λ1_max > λ_max
    # if λ1_max > λ_max:
    #     λ1_max_list.append(λ1_max)
    #     λ_max_list.append(λ_max)
    #     valid_indices1.append(i)

# Convert to NumPy arrays
s_X1_array = np.array(s_X1_list).flatten()
s_X_array = np.array(s_X_list).flatten()

# λ1_max_array = np.array(λ1_max_list).flatten()
# λ_max_array = np.array(λ_max_list).flatten()

# Check if we have valid data to plot
if len(s_X1_array) == 0:
    print("No trials where s_X1 > s_X; nothing to plot.")
else:
    # Plot configuration
    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet}\boldmath' 

    # Bar plot
    x = np.arange(len(s_X1_array))  # index of valid trials
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, s_X1_array, width, label=r'Data and Side information Conformant model $s_{\mathcal{X}}$', color='tab:orange')
    bars2 = ax.bar(x + width/2, s_X_array, width, label=r'Conformant model $s_{\mathcal{X}}$', color='tab:blue')

    # Labeling
    ax.set_xlabel('Trial Index')
    ax.set_ylabel(r'State Set Factor $s_{\mathcal{X}}$')
    ax.set_xticks(x)  
    ax.set_xticklabels([str(i + 1) for i in range(len(s_X1_array))])  # Start from 1
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Display plot
    plt.tight_layout()
    plt.show()