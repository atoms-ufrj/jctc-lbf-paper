
#-----------------------------------------------------------------------------------------
# PACKAGES 

import mics as mx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from simtk import unit

#-----------------------------------------------------------------------------------------
# SWITCHING FUNCTIONS

def H(x, A, B, C):
    return A*x**4 + B*x**3 + C*x**2 + (1-A-B-C)*x

def Hp(x, A, B, C): #derivative of H(x)
    return 4*A*x**3 + 3*B*x**2 + 2*C*x + (1-A-B-C)
       
#-----------------------------------------------------------------------------------------
# INPUTS

solute = 'phenol'
solvent = 'water'
T = 298.15*unit.kelvin 
kT = (unit.MOLAR_GAS_CONSTANT_R*T).value_in_unit(unit.kilojoules_per_mole)
lambdas_vdw = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]) 
A = 1.62
B = -0.889
C = 0.0255
nstates = 16
npoints = 201 
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

results.to_csv(f'{solute}-vdw-capped-initial.csv')

#-----------------------------------------------------------------------------------------
# REWEIGHTING

variables = dict(lbda=[], hC=[], hpC=[])
for point in range(npoints):
    lbda = point/(npoints - 1)
    variables['lbda'].append(lbda)
    variables['hC'].append(H(lbda, A, B, C))
    variables['hpC'].append(Hp(lbda, A, B, C))

properties = dict(
    Up='hpC*U_BG' 
    )

combinations = dict(
    F='kT*f'   
    )

reweighting = mixture.reweighting(
    potential=f'beta*({U})',
    properties=properties,
    combinations=combinations,
    conditions=pd.DataFrame(variables), 
    beta=1/kT, 
    kT=kT
    )

reweighting.to_csv(f'{solute}-vdw-capped-reweighting.csv')

#-----------------------------------------------------------------------------------------
# FINAL FREE ENERGY

deltaG = reweighting['F'].iloc[-1]

print(f'Delta G = {deltaG} kJ/mol')

#-----------------------------------------------------------------------------------------
# FIGURES

fig, ax = plt.subplots(2, 1, figsize=(5.0, 10.0),  dpi=300)

ax[0].plot(reweighting['lbda'], reweighting['F'], 'b-', label='Interpolated')
ax[0].plot(results['lbda'], results['F'], 'bo', label='Simulated')
ax[1].plot(reweighting['lbda'], reweighting['Up'], 'b-', label='')
ax[0].set_ylabel('$G$ (kJ/mol)')
ax[1].set_ylabel('$\partial G/ \partial \lambda$ (kJ/mol)')
ax[1].set_xlabel('$\lambda$')
ax[0].legend()
plt.rcParams['font.size'] = 15

plt.show()
#-----------------------------------------------------------------------------------------
