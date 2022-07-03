
#-----------------------------------------------------------------------------------------
# PACKAGES 

import mics as mx
import numpy as np
import pandas as pd
from simtk import unit
import scipy as sp
import scipy.integrate as integrate
  
#-----------------------------------------------------------------------------------------
# SWITCHING FUNCTIONS

def H(x, A, B, C):
    return A*x**4 + B*x**3 + C*x**2 + (1-A-B-C)*x

def Hp(x, A, B, C):  #derivative of H(x)
    return 4*A*x**3 + 3*B*x**2 + 2*C*x + (1-A-B-C)  
  
#-----------------------------------------------------------------------------------------
# INPUTS 

solute = 'phenol'
solvent = 'water'
nstates = 16 
T = 298.15*unit.kelvin 
kT = (unit.MOLAR_GAS_CONSTANT_R*T).value_in_unit(unit.kilojoules_per_mole)
lambdas_vdw = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]) 
A = 1.62
B = -0.889
C = 0.0255
U = 'hC*U_BG' 

#-----------------------------------------------------------------------------------------
# POTENTIAL AND FREE ENERGY OF THE SAMPLED STATES

mx.verbose = True
samples = mx.pooledsample()
for state in range(nstates):
    print(f'state {state}')
    lbda = lambdas_vdw[state]
    data = pd.read_csv(f'{solute}-vdw-capped-{int(100*lbda):03d}.csv')
    data.drop(index=range(20000), inplace=True)
    data.index = np.arange(0, len(data))
    kwargs = {'lbda': lbda, 'beta': 1.0/kT}
    kwargs['hC'] = H(lbda, A, B, C)
    samples.append(mx.sample(data, f'beta*({U})', acfun='U_BG', **kwargs))

samples.subsampling(integratedACF=False)
mixture = mx.mixture(samples, engine=mx.MBAR())
results = mixture.free_energies()
results['F'] = results['f']*kT
results['dF'] = results['df']*kT
#-----------------------------------------------------------------------------------------
# OBJECTIVE FUNCTION

mx.verbose = False

def reweighting(lbda, x):
    A = x[0]
    B = x[1]
    C = x[2]

    print(A, B, C)
            
    variables = dict(lbda=[], hC=[], hpC=[])
    for i in list(range(0,len(lbda))):
        variables['lbda'].append(lbda[i])
        variables['hC'].append(H(lbda[i], A, B, C))
        variables['hpC'].append(Hp(lbda[i], A, B, C))

    
    properties = dict(
    		Up = 'hpC*U_BG'
    		)
    
    rw = mixture.reweighting(
                  potential=f'beta*({U})',
                  properties=properties,
                  conditions=pd.DataFrame(variables), 
                  beta=1/kT
                  )
        
    return (rw['dUp'].values)**2

    
def objective(x):   
    f_obj = integrate.fixed_quad(reweighting, 0.0, 1.0, args=(x,), n=400)
    print(f_obj[0])
    return f_obj[0]

#-----------------------------------------------------------------------------------------
# OPTIMIZATION METHOD

x0 = [A, B, C]
bnds = ((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0))

res = sp.optimize.minimize(
    objective,
    x0,
    method='Powell',
    bounds=bnds,
    options={'xtol': 1e-6, 'ftol': 1e-6}
    )

print(res)

#-----------------------------------------------------------------------------------------


