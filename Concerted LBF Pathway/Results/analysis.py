
#-----------------------------------------------------------------------------------------
# PACKAGES 

import mics as mx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
npoints = 201 
U = 'hC*U_BG + hD*(U_LJ-U_BG) + hE*U_SC'

#-----------------------------------------------------------------------------------------
# POTENTIAL AND FREE ENERGY OF THE SAMPLED STATES

mx.verbose = True
samples = mx.pooledsample()
for state in range(nstates):
    print(f'state {state}')
    lbda = state/(nstates-1)
    data = pd.read_csv(f'{solute}-{int(100*lbda):03d}.csv')
    data['U_PME'] = data['Energy[0] (kJ/mole)'] - data['Potential Energy (kJ/mole)']
    data.drop(index=range(20000), inplace=True)
    data.index = np.arange(0, len(data))
    kwargs = {'lbda': lbda, 'beta': 1.0/kT}
    for a in ['C', 'D', 'E']:
        kwargs[f'h{a}'] = S((lbda - start[a])/(finish[a] - start[a]))
    samples.append(mx.sample(data, f'beta*({U})', acfun='U_BG', **kwargs))

samples.subsampling(integratedACF=False)
mixture = mx.mixture(samples, engine=mx.MBAR())
results = mixture.free_energies()
results['F'] = results['f']*kT
results['dF'] = results['df']*kT

results.to_csv(f'{solute}-{solvent}-initial.csv')

#-----------------------------------------------------------------------------------------
# REWEIGHTING

variables = dict(lbda=[], hC=[], hD=[], hE=[], hpC=[], hpD=[], hpE=[])
for point in range(npoints):
    lbda = point/(npoints - 1)
    variables['lbda'].append(lbda)
    for a in ['C', 'D', 'E']:
        y = (lbda - start[a])/(finish[a] - start[a])
        variables[f'h{a}'].append(S(y))
        variables[f'hp{a}'].append(Sp(y)*(1/(finish[a] - start[a])))

properties = dict(
    W='exp(-beta*(U_PME - U_SC))',
    Up='hpC*U_BG + hpD*(U_LJ-U_BG) + hpE*U_SC' 
    )

combinations = dict(
    F='kT*f',
    Fcorr='kT*(f - log(W))'    
    )

reweighting = mixture.reweighting(
    potential=f'beta*({U})',
    properties=properties,
    combinations=combinations,
    conditions=pd.DataFrame(variables), 
    beta=1/kT,
    kT=kT
    )

reweighting.to_csv(f'{solute}-{solvent}-reweighting.csv')

#-----------------------------------------------------------------------------------------
# FINAL FREE ENERGY

deltaG1 = reweighting['F'].iloc[-1]
deltaG2 = reweighting['Fcorr'].iloc[-1]

print(f'Delta G_CRF = {deltaG1} kJ/mol')
print(f'Delta G_PME = {deltaG2} kJ/mol')

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
