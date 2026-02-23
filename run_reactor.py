import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from reactor import *

# Define constant input signals
u = lambda t, x: [0.11, 300]

# Define intitial conditions
x0 = [5.0, 0.7, 270.0]

# Define parameters
p = reactor_default_parameters()

# Solve ODE with stiff solver
t_span = (0, 60.0*5.0)
t_eval = np.linspace(t_span[0], t_span[1], 300)
sol = sp.integrate.solve_ivp(reactor_ode, t_span, x0, args=(p, u), method='RK45', dense_output=True)

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(15,5))

axes[0].plot(t_eval, sol.sol(t_eval)[0])
axes[0].set_title('Height h(t)')
axes[1].plot(t_eval, sol.sol(t_eval)[0])
axes[1].set_title('Concentration c(t)')
axes[2].plot(t_eval, sol.sol(t_eval)[2])
axes[2].set_title('Temperature T(t)')
plt.show()