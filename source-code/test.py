#from  scipy import optimize
import scipy.optimize as sp
import numpy as np
def f(x):   # The rosenbrock function
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def fprime(x):
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))



print (sp.fmin_l_bfgs_b(f, [2, 2], fprime=fprime))
imp = __import__('importlib')
module  = imp.import_module ('scipy.optimize')
print (module)

