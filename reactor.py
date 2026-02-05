import numpy as np

def reactor_ode(t, x, p, u):
    """
    Defines the Ordinary Differential Equations for a CSTR reactor, based on Rawlings et al. (2017)
    Parameters:
    -----------
    t : float
        Current time
    x : list or np.array
        Current state vector [Height (h), Concentration (c), Temperature (T)]
    p : list or np.array
        Parameter vector [F0, T0, c0, r, k0, Er, U, rho, Cp, deltaH]
    u : list or np.array -- As functions of (x, t)
        Input vector [Fout, Tc]

    Returns:
    --------
    dxdt : list
        Derivatives [dh/dt, dc/dt, dT/dt]
    """

    # Unpack state variables
    h, c, T = x

    # Unpack parameters
    F0, T0, c0, r, k0, Er, U, rho, Cp, deltaH = p
    

    # Unpack inputs
    Fout, Tc = u(x, t)

    # ODEs
    dhdt = (F0 - Fout) / (np.pi * r**2)
    dcdt = (F0 * (c0 - c)) / (np.pi * r**2 * h) - k0 * c * np.exp(-Er/T)
    dTdt = (F0 * (T0 - T)) / (np.pi * r**2 * h) + (-deltaH / (rho * Cp)) * k0 * c * np.exp(-Er/T) + (2 * U * (Tc - T)) / (r * rho * Cp)

    return [dhdt, dcdt, dTdt]

def reactor_default_parameters():
    """
    Returns default parameter values for the CSTR reactor model, according to Rawlings et al. (2017).
    """
    F0 = 0.1 # m**3/min
    T0 = 350.0 # K
    c0 = 1 # kmol/m**3
    r = 0.219 # m
    k0 = 7.2e10 # 1/min
    Er = 8750.0 # K
    U = 54.94 # kJ/min*m**2*K
    rho = 1000.0 # kg/m**3
    Cp = 0.239 # kJ/kg*K
    deltaH = -5.0e4 # kJ/kmol
    return [F0, T0, c0, r, k0, Er, U, rho, Cp, deltaH]