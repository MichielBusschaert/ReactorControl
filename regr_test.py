import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# Generate data
f = lambda x: 4 + 2*x + 0.1*x**3
x = np.random.uniform(-10, 10, (20,))
xr = np.linspace(-10, 10, 1000)
y = f(x) + 0.5*np.random.normal(size=x.shape)
yr = f(xr)

# Regression model
regr = sk.linear_model.Lasso(max_iter=10000, fit_intercept=False, tol=1e-8)
clf = sk.model_selection.GridSearchCV(regr, {'alpha': np.logspace(-10, 0, 100)}, cv=10, refit=True)
X = np.hstack([x[:,None]**d for d in range(0, 6)])
Xr = np.hstack([xr[:,None]**d for d in range(0, 6)])
clf.fit(X, y)

# Visualize
fig = plt.figure()
plt.plot(xr, yr, 'k--', label="f(x)")
plt.scatter(x, y, marker='o', color='r', label="data")
plt.plot(xr, clf.predict(Xr), 'b-', label="regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig2 = plt.figure()
plt.semilogx(clf.cv_results_['param_alpha'], clf.cv_results_['mean_test_score'], 'o-')
plt.xlabel("alpha")
plt.ylabel("mean test score")
plt.show()