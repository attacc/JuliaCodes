import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction as frac
from matplotlib.ticker import FuncFormatter, MultipleLocator
import scipy.sparse.linalg as scp

def pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """Formats axis ticks with fractions of pi."""
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val / np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator

    if n == 0:
        return "$0$"
    elif d == 1:
        return f"${minus}{n}{pi}$"
    else:
        return f"${minus}\\frac{{{n}{pi}}}{{{d}}}$"

def e_k(t_0, vec_k, delta):
    """Calculates the kinetic energy term for the Hamiltonian."""
    ek = 0 + 0j
    for d in delta:
        dot_product = np.dot(vec_k, d).item()  # Convert to scalar
        ek += np.exp(1.0j * dot_product)
    return -t_0 * ek

def step(x):
    """Step function: returns 1 if x >= 0, otherwise 0."""
    return 1.0 if x >= 0 else 0.0

def kronecker_delta(p, q):
    """Kronecker delta function."""
    return 1.0 if p == q else 0.0

def simpson_rule(n, m, w, A, tau, Tot, T_spc, rule="3/8"):
    """  General implementation for Simpson's 1/3, 3/8, and Euler methods for numerical integration.  """

    # Assume uniform time spacing
    dt = np.abs(T_spc[1] - T_spc[0])  # Time step

    # Number of time steps
    n_steps = len(T_spc)

    # Set up rule-specific coefficients
    if rule == "1/3":
        step = 2  # Simpson's 1/3 rule requires an even number of intervals
        coeffs = [1, 4, 1]
        factor = dt / 3.0
    elif rule == "3/8":
        step = 3  # Simpson's 3/8 rule requires multiples of 3 intervals
        coeffs = [1, 3, 3, 1]
        factor = (3.0 / 8.0) * dt
    elif rule == "euler":
        coeffs = [1]  # Euler's method has no weighted sum, just one term
        factor = dt / Tot  # As per Euler method definition
    else:
        raise ValueError("Unsupported rule. Use '1/3', '3/8' or 'euler'.")

    # Initialize integral result and temporary storage for function values
    I = np.zeros_like(V_time(w, A, tau, Tot, T_spc[0]))  # Assuming V_time returns a numpy array
    f = np.zeros((len(coeffs), *I.shape), dtype=np.cdouble)  # Shape matches `V_time` outputs

    if rule == "euler":
        # Euler's method (requires computing V_time with exponential)
        Vt = 0.0
        for t in T_spc:
            Vt += V_time(w, A, tau, Tot, t) * np.exp(1j * w * (n - m) * t) * dt
        I = Vt
    else:
        # Perform integration using Simpson's rule (1/3 or 3/8)
        for j in range((n_steps - 1) // step):
            for k, coeff in enumerate(coeffs):
                idx = step * j + k
                # Extrapolate the time and V_time if necessary
                if idx < n_steps:
                    t_current = T_spc[idx]
                else:
                    # Extrapolate time by adding dt
                    t_current = T_spc[-1] + dt

                # Compute V_time with the complex exponential factor
                f[k] = V_time(w, A, tau, Tot, t_current) * np.exp(1j * w * (n - m) * t_current)

            # Add the weighted sum of the function evaluations to the integral
            I += sum(coeff * f_val for coeff, f_val in zip(coeffs, f)) * factor

    return I

def print_formatted_matrix(matrix):
    """
    Prints a 2D matrix in the specified format.
    Args:
        matrix (list or numpy.ndarray): A 2D matrix.
    """
    # Ensure matrix is a numpy array for consistency in handling
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    rows, cols = matrix.shape

    # Print the header row
    print("    i |" + "".join(f"   {col:^4}   |" for col in range(cols)))

    # Print the separator line
    print("  j   |" + "-" * (9 * cols))

    # Print each row with indices
    for i in range(rows):
        formatted_row = " | ".join(f"{matrix[i, j]:.3f}" for j in range(cols))
        print(f"  {i:<3} | {formatted_row} |")

def find_pairs(array):
    """
    Finds pairs of indices (i, j) in the array where array[j] is the negative of array[i].
    Identifies degenerate pairs and returns unique pairs including one from each degenerate group.
    """
    pairs = []
    used_indices = set()  # To track already paired indices
    abs_value_map = {}    # To group pairs by absolute values

    for i, value in enumerate(array):
        if i in used_indices:
            continue

        # Find the negative counterpart
        for j in range(i + 1, len(array)):
            if j in used_indices:
                continue

            if np.allclose(array[j], -value):
                pair = [i, j]
                pairs.append(pair)
                used_indices.add(i)
                used_indices.add(j)

                # Group by absolute value
                abs_val = round(abs(value), 8)  # Round for numerical stability
                if abs_val not in abs_value_map:
                    abs_value_map[abs_val] = []
                abs_value_map[abs_val].append(pair)
                break

    # Separate degenerate and unique pairs
    degenerate_pairs = []
    unique_pairs = []
    for abs_val, grouped_pairs in abs_value_map.items():
        if len(grouped_pairs) > 1:  # More than one pair for the same absolute value
            degenerate_pairs.append(grouped_pairs)
            unique_pairs.append(grouped_pairs[0])  # Include one representative
        else:
            unique_pairs.extend(grouped_pairs)

    return np.array(degenerate_pairs), np.array(unique_pairs)

def A_vector(w, A, tau, Tot, t):
   return A * (np.sin(np.pi * (t - tau) / Tot))**2 * np.cos(w * t) * step(Tot - (t - tau)) * step(t - tau)

def V_time(w, A, tau, Tot, t):
    """Calculates the time-dependent perturbation."""
    #sx = np.array([[0, 1], [1, 0]])
    #return (5.0+2.3j)*sx
    Vt = Tot_Ham(params, vec_k, w, A, Tot, t, tau)-Ham(params, vec_k)
    return Vt
    #return A * (np.sin(np.pi * (t - tau) / Tot))**2 * np.cos(w * t) * step(Tot - (t - tau)) * step(t - tau) * sx

def Ham(params, vec_k):
    """Constructs the static Hamiltonian for given parameters and momentum."""
    hdim, E_g, t_0, delta = params[0], params[3], params[4], params[5]
    H = np.zeros((hdim, hdim), dtype=np.cdouble)

    ek = e_k(t_0, vec_k, delta)
    for i in range(hdim):
      for j in range(hdim):
        if(i==j):
          H[i, j] = (-1)**i * E_g / 2.0
        elif i < j:
          H[i, j] = ek
        else:
          H[i, j] = np.conjugate(H[j, i])

    if not np.allclose(H, H.conj().T):
        raise ValueError("The Unperturbed Hamiltonian is not Hermitian!")

    return H

def Tot_Ham(params, vec_k, w, A, Tot, t,tau=0, qe=-1):
    """Constructs the full time dependent Hamiltonian for given parameters and momentum."""
    hdim, E_g, t_0, delta = params[0], params[3], params[4], params[5]
    H = np.zeros((hdim, hdim), dtype=np.cdouble)
    for i in range(hdim):
      for j in range(hdim):
        pierls_k = vec_k + qe*A_vector(w, A, tau, Tot, t)*(i-j)
        if(i==j):
          H[i, j] = (-1)**i * E_g / 2.0
        elif i < j:
          H[i, j] = e_k(t_0, pierls_k, delta)
        else:
          H[i, j] = np.conjugate(H[j, i])

    if not np.allclose(H, H.conj().T):
        raise ValueError("The Unperturbed Hamiltonian is not Hermitian!")

    return H

def Floq_Ham(F_modes,params, vec_k):
  hdim, A, w, Tot, Nm = params[0], params[6], params[7], params[8], params[11]
  H_F = np.zeros((Nm,Nm,hdim, hdim), dtype=np.cdouble)
  H_F_comb = np.zeros((Nm*hdim, Nm*hdim), dtype=np.cdouble)
  H = Ham(params, vec_k)
  #Vt = V_time(w, A, 0.0, Tot, 0.0)
  #Euler Integral
  '''Vt = simpson_rule(n, m, w, A, 0.0, Tot,rule="euler") / Tot'''
  #Simpson's 1/3 integral
  '''Vt = simpson_rule(n, m, w, A, 0.0, Tot,rule="1/3") / Tot'''
  #Simpson's 3/8 integral
  for i in range(Nm):
    n = F_modes[i]
    for j in range(Nm):
      m = F_modes[j]
      Vt = simpson_rule(n, m, w, A, 0.0, Tot,T_spc,rule="3/8") / Tot
      if(n==m):
        H_F[i,j,:,:]=H-(n*w)*np.eye(hdim)
      if(abs(n-m)==1):
        if n < m:
            H_F[i,j,:,:] = Vt
        else:
            H_F[i,j,:,:] = np.conjugate(H_F[j, i,:,:])

  H_F_comb = H_F.transpose(0, 2, 1, 3).reshape(Nm*hdim, Nm*hdim)
  if not np.allclose(H_F_comb, H_F_comb.conj().T):
        raise ValueError("The Floquet Hamiltonian is not Hermitian!")
  return H_F_comb

def Parallel_Ham(F_modes,params, k_space):
  hdim, Nk, Nm = params[0], params[9], params[11]
  H_F = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
  E_F = np.zeros((Nk,Nm*hdim), dtype=float)
  V_F = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
  for kp,vec_k in enumerate(k_space):
    H_F[kp,:,:] = Floq_Ham(F_modes,params, vec_k)
    E_F[kp,:], V_F[kp,:,:] = np.linalg.eigh(H_F[kp,:,:])
    formatted_eigenvalues = " | ".join(f"{x:.3f}" for x in E_F[kp, :])
    print(f"kp={kp} and vec_k={vec_k}\nEigenvalues:\n{formatted_eigenvalues}")
    print('Hamiltonian:')
    print_formatted_matrix(H_F[kp,:,:])
    print('Eigenvectors:')
    print_formatted_matrix(V_F[kp,:,:])
    #V_F_flat[kp,:,:,:,:] = V_F[kp,:,:].reshape(Nm,hdim, Nm,hdim).transpose(0, 2, 1, 3)
  return E_F, V_F


def Tot_Ham_new(params, vec_k, w, A, Tot, t,tau=0, qe=-1):
    """Constructs the full time dependent Hamiltonian for given parameters and momentum."""
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, 1]])
    E_g, t_0 = params[3], params[4]
    H = t_0*(np.cos(vec_k+qe*A*np.sin(w*t))*sx+(E_g / 2.0)*sz)
    return H

def V_time_new(w, A, tau, Tot, t):
    """Calculates the time-dependent perturbation."""
    #sx = np.array([[0, 1], [1, 0]])
    #return (5.0+2.3j)*sx
    Vt = Tot_Ham_new(params, vec_k, w, A, Tot, t, tau)-Ham(params, vec_k)
    return Vt

def simpson_rule_new(n, m, w, A, tau, Tot, T_spc, rule="3/8"):
    """  General implementation for Simpson's 1/3, 3/8, and Euler methods for numerical integration.  """

    # Assume uniform time spacing
    dt = np.abs(T_spc[1] - T_spc[0])  # Time step

    # Number of time steps
    n_steps = len(T_spc)

    # Set up rule-specific coefficients
    if rule == "1/3":
        step = 2  # Simpson's 1/3 rule requires an even number of intervals
        coeffs = [1, 4, 1]
        factor = dt / 3.0
    elif rule == "3/8":
        step = 3  # Simpson's 3/8 rule requires multiples of 3 intervals
        coeffs = [1, 3, 3, 1]
        factor = (3.0 / 8.0) * dt
    elif rule == "euler":
        coeffs = [1]  # Euler's method has no weighted sum, just one term
        factor = dt / Tot  # As per Euler method definition
    else:
        raise ValueError("Unsupported rule. Use '1/3', '3/8' or 'euler'.")

    # Initialize integral result and temporary storage for function values
    I = np.zeros_like(V_time_new(w, A, tau, Tot, T_spc[0]))  # Assuming V_time returns a numpy array
    f = np.zeros((len(coeffs), *I.shape), dtype=np.cdouble)  # Shape matches `V_time` outputs

    if rule == "euler":
        # Euler's method (requires computing V_time with exponential)
        Vt = 0.0
        for t in T_spc:
            Vt += V_time_new(w, A, tau, Tot, t) * np.exp(1j * w * (n - m) * t) * dt
        I = Vt
    else:
        # Perform integration using Simpson's rule (1/3 or 3/8)
        for j in range((n_steps - 1) // step):
            for k, coeff in enumerate(coeffs):
                idx = step * j + k
                # Extrapolate the time and V_time if necessary
                if idx < n_steps:
                    t_current = T_spc[idx]
                else:
                    # Extrapolate time by adding dt
                    t_current = T_spc[-1] + dt

                # Compute V_time with the complex exponential factor
                f[k] = V_time_new(w, A, tau, Tot, t_current) * np.exp(1j * w * (n - m) * t_current)

            # Add the weighted sum of the function evaluations to the integral
            I += sum(coeff * f_val for coeff, f_val in zip(coeffs, f)) * factor

    return I

def Floq_Ham_new(F_modes,params, vec_k):
  hdim, A, w, Tot, Nm = params[0], params[6], params[7], params[8], params[11]
  H_F = np.zeros((Nm,Nm,hdim, hdim), dtype=np.cdouble)
  H_F_comb = np.zeros((Nm*hdim, Nm*hdim), dtype=np.cdouble)
  H = Ham(params, vec_k)
  #Vt = V_time(w, A, 0.0, Tot, 0.0)
  #Euler Integral
  #Vt = simpson_rule(n, m, w, A, 0.0, Tot,rule="euler") / Tot
  #Simpson's 1/3 integral
  #Vt = simpson_rule(n, m, w, A, 0.0, Tot,rule="1/3") / Tot
  #Simpson's 3/8 integral
  for i in range(Nm):
    n = F_modes[i]
    for j in range(Nm):
      m = F_modes[j]
      Vt = simpson_rule_new(n, m, w, A, 0.0, Tot,T_spc,rule="3/8") / Tot
      if(n==m):
        H_F[i,j,:,:]=H-(n*w)*np.eye(hdim)
      if(abs(n-m)==1):
        if n < m:
            H_F[i,j,:,:] = Vt
        else:
            H_F[i,j,:,:] = np.conjugate(H_F[j, i,:,:])

  H_F_comb = H_F.transpose(0, 2, 1, 3).reshape(Nm*hdim, Nm*hdim)
  if not np.allclose(H_F_comb, H_F_comb.conj().T):
        raise ValueError("The Floquet Hamiltonian is not Hermitian!")
  return H_F_comb

def Parallel_Ham_new(F_modes,params, k_space):
  hdim, Nk, Nm = params[0], params[9], params[11]
  H_F = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
  E_F = np.zeros((Nk,Nm*hdim), dtype=float)
  V_F = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
  for kp,vec_k in enumerate(k_space):
    H_F[kp,:,:] = Floq_Ham_new(F_modes,params, vec_k)
    E_F[kp,:], V_F[kp,:,:] = np.linalg.eigh(H_F[kp,:,:])
    formatted_eigenvalues = " | ".join(f"{x:.3f}" for x in E_F[kp, :])
    print(f"kp={kp} and vec_k={vec_k}\nEigenvalues:\n{formatted_eigenvalues}")
    print('Hamiltonian:')
    print_formatted_matrix(H_F[kp,:,:])
    print('Eigenvectors:')
    print_formatted_matrix(V_F[kp,:,:])
    #V_F_flat[kp,:,:,:,:] = V_F[kp,:,:].reshape(Nm,hdim, Nm,hdim).transpose(0, 2, 1, 3)
  return E_F, V_F


# Parameters and initialization
sdim = 1
hdim = 2
a_cc = 1.0
t_0 = 1.0
E_g = 0.0
Nk = 101
Nt = 100
mod_lim = 2
Nm = 2*mod_lim+1
w = 0.25 #np.pi
A = 0.5 #100
Tot = (2 * np.pi) / w
my_dpi = 700

T_spc = np.linspace(0, 1.5 * Tot, Nt)
F_modes = np.arange(-mod_lim, mod_lim + 1)

delta = np.array([[a_cc], [-a_cc]])
         #  0    1     2     3    4     5    6  7   8   9   10  11    12
params = [hdim, sdim, a_cc, E_g, t_0, delta, A, w, Tot, Nk, Nt, Nm, mod_lim]
k_space = np.linspace(-np.pi / 2.0, np.pi / 2.0, Nk)

color_map = ['red', 'blue', 'violet', 'cyan', 'pink','royalblue', 'maroon', 'teal', 'salmon', 'deepskyblue']
color_map_pos = ['blue', 'cyan', 'royalblue', 'teal', 'deepskyblue']
color_map_neg = ['red', 'violet', 'pink', 'maroon', 'salmon']

H = np.zeros((Nk, hdim, hdim), dtype=np.cdouble)
E = np.zeros((Nk, hdim), dtype=float)
V = np.zeros((Nk, hdim, hdim), dtype=np.cdouble)

VF = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
EF = np.zeros((Nk,Nm*hdim), dtype=float)

for kp,vec_k in enumerate(k_space):
  H[kp,:,:] = Ham(params, vec_k)
  E[kp,:], V[kp,:,:] = np.linalg.eigh(H[kp,:,:])
  for a in range(hdim):
    plt.scatter(vec_k,E[kp,a], color=color_map[a])
plt.savefig('1D-BS.png',dpi=my_dpi, bbox_inches='tight')
plt.close()
#H_F=Floq_Ham(F_modes,params, vec_k)


W_set = [1.0,0.25]
F_set = [0, 0.5]
Char = ['a','b','d','e']
y_lim = [1,0.5,1.5,0.125]
x_lim1 = [0,0,0,np.pi/3.0]
x_lim2 = [np.pi/3.0, np.pi/3.0,np.pi/1.5,np.pi/1.5]
k_space = np.linspace(0, np.pi, Nk)
E_g = 0.2
ind = 0
for w in W_set:
  for F in F_set:
    A = F*w

    params = [hdim, sdim, a_cc, E_g, t_0, delta, A, w, Tot, Nk, Nt, Nm, mod_lim]
    
    EF,VF = Parallel_Ham_new(F_modes,params, k_space)
    #EF,VF = Parallel_Ham(F_modes,params, k_space)

    for a in range(Nm*hdim):
        plt.plot(k_space,EF[:,a], label=f'Column-{a}')
    plt.ylim(-y_lim[ind],y_lim[ind])
    plt.xlim(x_lim1[ind],x_lim2[ind])

    # Set x-axis tick locations and labels
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 6))  # Ticks at intervals of pi/2
    ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))  # Format ticks as fractions of pi

    plt.legend()
    plt.title(f"For F = {F} and $\\Omega$ = {w}")
    plt.savefig(f'1D-BS-Floquet_{Char[ind]}[new].png',dpi=my_dpi, bbox_inches='tight')
    plt.close()
    ind+=1
#H_F=Floq_Ham(F_modes,params, vec_k)