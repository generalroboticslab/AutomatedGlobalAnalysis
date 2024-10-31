import numpy as np

def autapse(t, x):
    w = .5
    return -x + np.tanh(w*x)

def simple_koop(t, x):
    x1 = x[0]
    x2 = x[1]
    mu = -0.1
    lam = -1
    return [mu*x1, lam*(x2 - x1**2)]

def double_lco(t, cart):
    x, y = cart
    r = np.sqrt(x**2 + y**2)
    dot_x = x * (1/3 - r) * (2/3 - r) * (1 - r) + y
    dot_y = y * (1/3 - r) * (2/3 - r) * (1 - r) - x
    return [dot_x, dot_y]

def double_pend(t, thetas):
    L1 = 1
    L2 = 1
    m1 = 1
    m2 = 2
    beta = 0.1
    g = 9.81

    theta1, theta1_dot, theta2, theta2_dot = thetas
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    
    theta1_dot = theta1_dot
    theta1_dot_dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*theta1_dot**2*c + L2*theta2_dot**2) - (m1+m2)*g*np.sin(theta1) - beta*theta1_dot) / L1 / (m1 + m2*s**2)
    theta2_dot = theta2_dot
    theta2_dot_dot = ((m1+m2)*(L1*theta1_dot**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + m2*L2*theta2_dot**2*s*c - beta*theta2_dot) / L2 / (m1 + m2*s**2)
    
    return theta1_dot, theta1_dot_dot, theta2_dot, theta2_dot_dot

def duffing(t, x):
    alpha = 1.0
    beta = -1.0
    delta = 0.5
    dxdt = [x[1], -delta*x[1] - beta*x[0] - alpha*x[0]**3]
    return dxdt

# Constants for the Hodgkin-Huxley model
C_m = 1.0  # Membrane capacitance, in uF/cm^2
g_Na = 120.0  # Maximum conductances, in mS/cm^2
g_K = 36.0
g_L = 0.3
E_Na = 50.0  # Equilibrium potentials, in mV
E_K = -77.0
E_L = -54.387
I_ext = 10.0  # External current in uA/cm^2

# Rate functions
def alpha_m(V): return 0.1 * (V + 45) / (1 - np.exp(-(V + 45) / 10))
def beta_m(V): return 4.5 * np.exp(-(V + 70) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 70) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 40) / 10))
def alpha_n(V): return 0.01 * (V + 60) / (1 - np.exp(-(V + 60) / 10))
def beta_n(V): return 0.15 * np.exp(-(V + 70) / 80)

# The Hodgkin-Huxley model differential equations
def hodgkin_huxley(t, y):
    V, m, h, n = y
    dVdt = (I_ext - g_Na * m**3 * h * (V - E_Na) - g_K * n**4 * (V - E_K) - g_L * (V - E_L)) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return dVdt, dmdt, dhdt, dndt


def pend(t, thetas):
    g = 9.81
    l = 1
    beta = 0.2
    theta, theta_dot = thetas

    theta_ddot = -g/l*np.sin(theta)-beta*theta_dot
    return [theta_dot, theta_ddot]

def linear_magnet_model(t, x):
    m = 1.0
    c = 0.5
    k = 10.0
    alpha = 100.0
    h = 1.5
    b = 1.3
    dxdt = [x[1], (1/m)*(-c*x[1] - k*x[0] + alpha*(x[0]-b)*(12*h**2-3*(x[0]-b)**2)*((x[0]-b)**2+h**2)**(-7/2))]
    return dxdt

def lorenz_96(t, xs):
    """Lorenz 96 dynamical system."""
    grid_size = 40
    F = 2.75
    dxdt = np.zeros(grid_size)

    def dx_dt(xs, j):
        """Lorenz 96 differential equations."""

        jm1 = (j - 1) % grid_size
        jm2 = (j - 2) % grid_size
        jp1 = (j + 1) % grid_size
        return ((xs[jp1] - xs[jm2]) * xs[jm1]) - xs[j] + F

    for j in range(grid_size):
        dxdt[j] = dx_dt(xs, j)

    return dxdt

def tri_stable_dynamics(t, x):
    alpha = 20
    beta = 8
    gamma = 6/10
    delta = .8
    x_dot = x[1]
    x_dot_dot = -alpha*x[0] + beta*x[0]**3 - gamma*x[0]**5 - delta*x[1]
    return np.array([x_dot, x_dot_dot])

def van_der_pol(t, x):
    mu = .2
    dxdt = [x[1], mu*(1 - x[0]**2)*x[1] - x[0]]
    return dxdt