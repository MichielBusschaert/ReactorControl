import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from vanderpol_sysid import *

# Define Van der Pol Oscillator Parameters
def vdpode(t, y, mu):
    dydt = np.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = mu*(1-y[0]**2)*y[1] - y[0]
    return dydt

# Define vector fields
mu = 0.5
u = np.linspace(-3, 3, 20)
v = np.linspace(-3, 3, 20)
U = np.zeros((len(u), len(v)))
V = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        dydt = vdpode(0, [u[i], v[j]], mu)
        U[i,j] = dydt[0]
        V[i,j] = dydt[1]

# Plot a trajectory
t_span = (0, 30)
dt = 0.1
t_eval = np.arange(t_span[0], t_span[1], dt)
y0 = np.array([1.0, 0.0])
sol = sp.integrate.solve_ivp(vdpode, t_span, y0, args=(mu,), rtol=1e-8, atol=1e-12, t_eval=t_eval)

# Plot vector field
plt.figure(figsize=(10, 10))
plt.quiver(u, v, U, V, color='r')
plt.plot(sol.y[0], sol.y[1], 'b')
plt.title('Van der Pol Oscillator Vector Field')

# Generate data for SysID
y0_box = np.array([[-3, 3], [-3, 3]])
train_sols, test_sols = GenerateData(vdpode, t_span, dt, y0_box, Nsamples=50, args=(mu,), train_per=0.75)

# Compute DMD decomposition
L, E, D = DynamicModeDecomposition(train_sols)
dmd_sols = [PredictDMD(L, E, D, test_sols[i].y[:,0], dt, t_span) for i in range(len(test_sols))]

# Plot DMD predictions
plt.plot(dmd_sols[0][1][0,:], dmd_sols[0][1][1,:], 'g--')
plt.plot(test_sols[0].y[0], test_sols[0].y[1], 'g')

# Compute SINDy
sindy, theta_library = SparseIdentificationNonlinearDynamics(train_sols)
sindy_sols = [PredictSINDy(sindy, theta_library, test_sols[i].y[:,0], dt, t_span) for i in range(len(test_sols))]

# Plot SINDy predictions
plt.plot(sindy_sols[0][1][0,:], sindy_sols[0][1][1,:], 'c--')
plt.plot(test_sols[0].y[0], test_sols[0].y[1], 'g')

# Compute eDMD
L, E, D, C, psi = ExtendedDynamicModeDecomposition(train_sols)
edmd_sols = [PredictEDMD(L, E, D, C, psi, test_sols[i].y[:,0], dt, t_span) for i in range(len(test_sols))]

# Plot eDMD predictions
plt.plot(edmd_sols[0][1][0,:], edmd_sols[0][1][1,:], 'm--')
plt.plot(test_sols[0].y[0], test_sols[0].y[1], 'g')