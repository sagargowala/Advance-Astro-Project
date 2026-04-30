import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd


#----- Eos and others assumptions -----
gamma = 5/3        # Adiabatic index (monatomic gas)
K = 0.1           # Adiabatic constant (P = K * rho^gamma)
C_q = 3.0         # Artificial viscosity coefficient
C_courant = 0.3    # Courant factor (< 1.0 for stability)
M_total = 1.0      # Total mass of the system

def initialise_uniform_density(N):
    # N Number of mass shells
    rho0= 3/(4*np.pi)         # average uniform density

    # --- 2. Grid Setup (Mass Coordinates) ---
    dm = M_total / N
    m_nodes = np.linspace(0, M_total, N + 1)      # Mass at shell boundaries
    m_centers = (m_nodes[1:] + m_nodes[:-1]) / 2  # Mass at shell centers
    r = (3* m_nodes / (4 * np.pi *rho0))**(1/3)  # Since M(r) = (4/3) * pi * r^3 * rho, then r = (3M / 4pi)^1/3
    v = np.zeros(N + 1)   # Initially at rest, velocity=0
    vol = 4/3 * np.pi * (r[1:]**3 - r[:-1]**3)
    rho = dm / vol
    p = K*rho**gamma
    return r, v, p, rho, dm, m_centers, m_nodes, N


def lame_emden_solver(nn=3,xi_max=7,NN=101):
    #first order system of equations
    def lane_emden(xi, Y):
        y1, y2 = Y
        dy1_dx=y2
        dy2_dx= -(2/xi)*y2 - y1**nn
        return [dy1_dx,dy2_dx]

    #initial conditions
    xi0 = 1e-6
    theta0 = 1 - (1/6)*xi0**2
    dtheta0 = - (1/3)*xi0
    Y0 = [theta0, dtheta0]

    xi_eval = np.linspace(xi0, xi_max, NN)
    sol = solve_ivp(lane_emden,(xi0, xi_max), Y0, t_eval=xi_eval, method='RK45')
    
    xi = sol.t
    theta = sol.y[0]
    theta_prime = sol.y[1]

    for ii in range(1, len(theta)):
        if theta[ii-1] > 0 and theta[ii] <= 0:
            x1, x2 = xi[ii-1], xi[ii]
            y1, y2 = theta[ii-1], theta[ii]
            v1, v2 = theta_prime[ii-1], theta_prime[ii]

            # interpolation factor
            t = -y1 / (y2 - y1)
            xi1 = x1 + t * (x2 - x1)
            theta_prime_1 = v1 + t * (v2 - v1)
    xi=xi[:ii]
    theta=theta[:ii]
    theta_prime=theta_prime[:ii]

    r =xi/xi1
    m = (xi**2 * theta_prime)/ (xi1**2 * theta_prime_1) 
    rho_desh = theta**nn
      
    return r, m , rho_desh


def initialise_white_dwarf(N):
    # N Number of mass shells
    r,m,_ = lame_emden_solver(nn=3,xi_max=7,NN=102)
    # --- 2. Grid Setup (Mass Coordinates) ---
    dm = np.diff(m)
    m_nodes = m      # Mass at shell boundaries
    m_centers = (m_nodes[1:] + m_nodes[:-1]) / 2  # Mass at shell centers
    v = np.zeros(N+1)   # Initially at rest, velocity=0
    vol = 4/3 * np.pi * (r[1:]**3 - r[:-1]**3)
    rho = dm / vol
    p = 0.99*K*rho**gamma
    return r, v, p, rho, dm, m_centers, m_nodes, N

def initialise_white_dwarf2(N):
    # lame_emden_solver returns r and m arrays
    r_raw, m_raw, _ = lame_emden_solver(nn=3, xi_max=7, NN=102)
    M_total = m_raw[-1]  # Total mass from the solver
    
    # --- 2. Grid Setup (Mass Coordinates) ---
    # Define the mass nodes for exactly N shells
    m_nodes = np.linspace(0, M_total, N + 1)
    m_centers = (m_nodes[1:] + m_nodes[:-1]) / 2
    dm = M_total / N  # Constant mass per shell in this setup
    
    # Interpolate radius from the Lane-Emden mass distribution
    # We map m_nodes (target) from m_raw (source) to get corresponding radii
    r = np.interp(m_nodes, m_raw, r_raw)
    
    # Initially at rest
    v = np.zeros(N + 1)   
    
    # --- 3. Thermodynamic Variables ---
    # Calculate volume of the shells defined by the interpolated radii
    vol = 4/3 * np.pi * (r[1:]**3 - r[:-1]**3)
    
    # Local density: mass of shell / volume of shell
    rho = dm / vol
    
    # Pressure from Polytropic Equation of State
    p = K * rho**gamma
    
    return r, v, p, rho, dm, m_centers, m_nodes, N


def collapse_simulation(initial_star_model, t_end):
    r, v, p, rho, dm, m_centers, m_nodes, N = initial_star_model()
    hist = {key: [] for key in ['t', 'r', 'v', 'p', 'rho', 'e', 'cs']}

    t = 0.0
    gamma_current = gamma 
    has_rebounded = False
    
    # Initialize r_old to compare against in the first loop
    r_old = r.copy()

    while t < t_end:
        # DETECT TURNING POINT
        # Check if the inner-most shell radius has started to increase
        # delta_r = r_now - r_prev
        if not has_rebounded and t > 0.01:
            if r[1] - r_old[1] > 0:  # The sign change (Turning Point)
                gamma_current = 2
                has_rebounded = True

        # Store current r to be r_old for the next iteration
        r_old = r.copy()

        # --- Physics calculations ---
        cs = np.sqrt(gamma_current * p / rho)
        dr = np.diff(r)
        
        # Courant-Friedrichs-Lewy (CFL) condition
        dt = C_courant * np.min(dr / (cs + np.abs(v[:-1] + v[1:])/2 + 1e-10))
        if t + dt > t_end: dt = t_end - t
        
        # Artificial Viscosity
        dv = np.diff(v)
        q = np.zeros_like(p)
        q[dv < 0] = (C_q * dr[dv < 0])**2 * rho[dv < 0] * (dv[dv < 0]/dr[dv < 0])**2
        
        # Momentum
        p_tot = p + q
        dp_dm = np.zeros(N + 1)
        dp_dm[1:-1] = np.diff(p_tot) / np.diff(m_centers)
        dp_dm[-1] = (0 - p_tot[-1]) / (M_total - m_centers[-1])
        
        accel = np.zeros_like(v)
        accel[1:] = -4 * np.pi * r[1:]**2 * dp_dm[1:] - (m_nodes[1:]) / r[1:]**2
        
        v += accel * dt
        v[0] = 0.0
        r = np.maximum(r + v * dt, 1e-10)
        
        # Thermodynamics (using the updated gamma)
        rho = dm / (4/3 * np.pi * (r[1:]**3 - r[:-1]**3))
        p = K * rho**gamma_current
        e = p / (rho * (gamma_current - 1))
        
        t += dt
        
        # Store for history
        hist['t'].append(t)
        hist['r'].append(r.copy())
        hist['v'].append(v.copy())
        hist['p'].append(p.copy())
        hist['rho'].append(rho.copy())
        hist['e'].append(e.copy())
        hist['cs'].append(cs.copy())
        
    t_arr = np.array(hist['t'])              # shape (Nt,)
    r_arr = np.array(hist['r'])              # shape (Nt, N+1)
    v_arr = np.array(hist['v'])              # shape (Nt, N+1)
    cs_arr = np.array(hist['cs'])            # shape (Nt, N)
    e_arr = np.array(hist['e'])              # shape (Nt, N)
    p_arr = np.array(hist['p'])              # shape (Nt, N)
    rho_arr = np.array(hist['rho'])          # shape (Nt, N)

    return t_arr, r_arr, v_arr, p_arr, rho_arr, e_arr, cs_arr  

#t_arr, r_arr, v_arr, p_arr, rho_arr, e_arr, cs_arr = collapse_simulation(lambda: initialise_white_dwarf(100), t_end=1.3)
