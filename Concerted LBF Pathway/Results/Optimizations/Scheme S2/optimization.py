
#-----------------------------------------------------------------------------------------
# PACKAGES 

import mics as mx
import numpy as np
import pandas as pd
import scipy as sp
import scipy.integrate as integrate
from simtk import unit

#-----------------------------------------------------------------------------------------
# SWITCHING FUNCTIONS

def S(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else (6*x**2 - 15*x + 10)*x**3)

def Sp(x): #derivative of S(x)
    return 0.0 if x < 0 else (0.0 if x > 1 else 30*(x**2 - 2*x + 1)*x**2)

#-----------------------------------------------------------------------------------------
# INPUTS 

solute = 'phenol'
solvent = 'water'
T = 298.15*unit.kelvin 
kT = (unit.MOLAR_GAS_CONSTANT_R*T).value_in_unit(unit.kilojoules_per_mole)
start  = dict(C=0.0, D=0.4, E=0.6) 
finish = dict(C=0.7, D=0.8, E=1.0) 
nstates = 21 
U = 'hC*U_BG + hD*(U_LJ-U_BG) + hE*U_SC'

#-----------------------------------------------------------------------------------------
# POTENTIAL AND FREE ENERGY OF THE SAMPLED STATES

mx.verbose = True
samples = mx.pooledsample()   
for state in range(nstates):
    print(f'state {state}')
    lbda = state/(nstates-1)
    data = pd.read_csv(f'{solute}-{int(100*lbda):03d}.csv')
    data.drop(index=range(20000), inplace=True)
    data.index = np.arange(0, len(data))
    kwargs = {'lbda': lbda, 'beta': 1.0/kT}
    for a in ['C', 'D', 'E']:
        kwargs[f'h{a}'] = S((lbda - start[a])/(finish[a] - start[a]))
    samples.append(mx.sample(data, f'beta*({U})', acfun='U_BG', **kwargs))    

samples.subsampling(integratedACF=False)
mixture = mx.mixture(samples, engine=mx.MBAR())
results = mixture.free_energies()

#-----------------------------------------------------------------------------------------
# OBJECTIVE FUNCTION

mx.verbose = False

def reweighting(lbda, x):
    C0 = 0.0
    C1 = x[0]
    E0 = D0 = x[1]*C1
    E1 = D1 = 1.0
    
    print(C1, D0)
        
    start  = dict(C=C0, D=D0, E=E0)
    finish = dict(C=C1, D=D1, E=E1)
    
    variables = dict(lbda=[], hC=[], hD=[], hE=[], hpC=[], hpD=[], hpE=[])
    for i in list(range(0,len(lbda))):
        variables['lbda'].append(lbda[i])
        for a in ['C', 'D', 'E']:
            y = (lbda[i] - start[a])/(finish[a] - start[a])
            variables[f'h{a}'].append(S(y))
            variables[f'hp{a}'].append(Sp(y)*(1/(finish[a] - start[a])))
    
    properties = dict(
    		Up = 'hpC*U_BG + hpD*(U_LJ-U_BG) + hpE*U_SC'
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

C1 = 0.7
D0 = 0.4

x0 = [C1, D0/C1]
bnds = ((0.0, 1.0), (0.0, 1.0))

res = sp.optimize.minimize(
    objective,
    x0,
    method='Powell',
    bounds=bnds,
    options={'xtol': 1e-6, 'ftol': 1e-6}
    )

print(res)



