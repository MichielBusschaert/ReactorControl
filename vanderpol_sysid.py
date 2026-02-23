import numpy as np
import scipy as sp
import sklearn as sk

def GenerateData(odefun, t_span, dt, y0_box, Nsamples, args=(), train_per=0.75):
    # Initialize
    t_eval = np.arange(t_span[0], t_span[1], dt)
    y0 = np.random.uniform(y0_box[:,0], y0_box[:,1], (Nsamples, y0_box.shape[0]))

    # Generate data
    ode_sols = []
    for i in range(Nsamples):
        sol = sp.integrate.solve_ivp(odefun, t_span, y0[i], args=args, t_eval=t_eval, rtol=1e-8, atol=1e-12, dense_output=True)
        ode_sols.append(sol)

    # Split into training and test sets
    randperm = np.random.permutation(Nsamples)
    train_sols = [ode_sols[i] for i in randperm[:round(train_per*Nsamples)]]
    test_sols = [ode_sols[i] for i in randperm[round(train_per*Nsamples):]]

    return (train_sols, test_sols)

def DynamicModeDecomposition(train_sols):
    # Put data in matrices
    X = np.hstack([ts.y[:,:-1] for ts in train_sols])
    Xprime = np.hstack([ts.y[:,1:] for ts in train_sols])

    # Compute SVD decomposition of training data
    U, S, Vh = np.linalg.svd(X, full_matrices=False, hermitian=False)

    # Compute reduced DMD matrix
    A = U.conj().T @ Xprime @ (Vh.conj().T / S[None, :])

    # Compute eigendecomposition of DMD matrix
    L, W = np.linalg.eig(A)

    # Compute encoding and decoding matrices
    E = np.linalg.solve(W, U.conj().T)
    D = U.conj().T @ W

    return L, E, D

def PredictDMD(L, E, D, y0, dt, t_span):
    # Initialize
    t = np.arange(t_span[0], t_span[1]+dt, dt)
    y = np.zeros((len(y0), len(t)))
    y[:,0] = y0

    # Predict trajectory
    z = E @ y0
    for i in range(1, len(t)):
        z = L * z
        y[:,i] = D @ z

    # Return output
    return t, y

def ExtendedDynamicModeDecomposition(train_sols, psi=lambda z: PolynomialLibrary(z, degree=2)):
    # Put data in matrices
    X = np.hstack([ts.y[:,:-1] for ts in train_sols])
    Xprime = np.hstack([ts.y[:,1:] for ts in train_sols])

    # Apply lifting functions to data --- Currently predefined, make more flexble later
    Psi = psi(X)
    PsiPrime = psi(Xprime)

    # Compute SVD decomposition of training data
    U, S, Vh = np.linalg.svd(Psi, full_matrices=False, hermitian=False)

    # Compute reduced DMD matrix
    A = U.conj().T @ PsiPrime @ (Vh.conj().T / S[None, :])
    C = X @ (Vh.conj().T / S[None, :]) @ U.conj().T

    # Compute eigendecomposition of DMD matrix
    L, W = np.linalg.eig(A)

    # Compute encoding and decoding matrices
    E = np.linalg.solve(W, U.conj().T)
    D = U.conj().T @ W

    return L, E, D, C, psi

def PredictEDMD(L, E, D, C, psi, y0, dt, t_span):
    # Initialize
    t = np.arange(t_span[0], t_span[1]+dt, dt)
    y = np.zeros((len(y0), len(t)))
    y[:,0] = y0

    # Predict trajectory
    z = (E @ psi(y0[:,None])).flatten()
    for i in range(1, len(t)):
        z = L * z
        y[:,i] = C @ (D @ z)

    # Return output
    return t, y

def SparseIdentificationNonlinearDynamics(train_sols, theta_library=lambda z: PolynomialLibrary(z, degree=4)):
    # Put data in matrices
    X = np.hstack([ts.y[:,:-1] for ts in train_sols])
    Xprime = np.hstack([ts.y[:,1:] for ts in train_sols])

    # Construct library of candidate functions --- Currently predefined, make more flexible later
    Theta = theta_library(X)

    # Perform sparse regression to identify governing equations
    lasso_regr = sk.linear_model.Lasso(random_state=None, max_iter=10000, fit_intercept=False)
    sindy = sk.model_selection.GridSearchCV(lasso_regr, {'alpha': np.logspace(-6, 0, 10)}, cv=10, refit=True)
    sindy.fit(Theta.T, Xprime.T)

    return sindy, theta_library

def PredictSINDy(sindy, theta_library, y0, dt, t_span):
    # Initialize
    t = np.arange(t_span[0], t_span[1]+dt, dt)
    y = np.zeros((len(y0), len(t)))
    y[:,0] = y0

    # Predict trajectory
    for i in range(1, len(t)):
        y[:,(i,)] = sindy.predict(theta_library(y[:,(i-1,)]).T).T

    # Return output
    return t, y

def PolynomialLibrary(X, degree=1):
    # Get dimensions
    n, m = X.shape # n: number of states, m: number of samples

    # Initialize library
    Theta = np.ones((1,m)) # Constant term
    Theta_deg = Theta # Copy

    for d in range(1,degree+1):
        Theta_deg = np.vstack([Theta_deg*X[i,:].reshape(1,-1) for i in range(n)])
        Theta = np.vstack([Theta, Theta_deg])

    return Theta