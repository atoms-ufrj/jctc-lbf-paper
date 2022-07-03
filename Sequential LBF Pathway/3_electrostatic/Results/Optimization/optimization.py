
#-----------------------------------------------------------------------------------------
# PACKAGES 

import mics as mx
import numpy as np
import pandas as pd
from simtk import unit
import scipy as sp
import scipy.integrate as integrate
    
#-----------------------------------------------------------------------------------------
# INPUTS 

solute = 'phenol'
solvente = 'water'
nstates = 5 
T = 298.15*unit.kelvin 
kT = (unit.MOLAR_GAS_CONSTANT_R*T).value_in_unit(unit.kilojoules_per_mole)
lambdas_coul = np.array([0.00, 0.25, 0.50, 0.75, 1.00]) 
U = 'hE*U_SC' 

#-----------------------------------------------------------------------------------------
# POTENTIAL AND FREE ENERGY OF THE SAMPLED STATES

mx.verbose = True
samples = mx.pooledsample()
for state in range(nstates):
    print(f'state {state}')
    lbda = lambdas_coul[state]
    data = pd.read_csv(f'{solute}-electrostatic-{int(100*lbda):03d}.csv')
    data.drop(index=range(20000), inplace=True)
    data.index = np.arange(0, len(data))
    kwargs = {'lbda': lbda, 'beta': 1.0/kT}
    kwargs['hE'] = lbda
    samples.append(mx.sample(data, f'beta*({U})', acfun='U_SC', **kwargs))

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
            
    variables = dict(lbda=[], hE=[], hpE=[])
    for i in list(range(0,len(lbda))):
        variables['lbda'].append(lbda[i])
        variables['hE'].append(A*(lbda[i])**4 + B*(lbda[i])**3 + C*(lbda[i])**2 + (1-A-B-C)*(lbda[i]))
        variables['hpE'].append(4*A*(lbda[i])**3 + 3*B*(lbda[i])**2 + 2*C*(lbda[i]) + (1-A-B-C))

    
    properties = dict(
    		Up = 'hpE*U_SC'
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

A = 0.0
B = 0.0
C = 0.0

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



