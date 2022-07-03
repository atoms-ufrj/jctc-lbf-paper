
#-----------------------------------------------------------------------------------------
# PACKAGES 

import mics as mx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from simtk import unit

#-----------------------------------------------------------------------------------------
# INPUTS

solute = 'phenol'
nstates = 5 
npoints = 201 
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
    data = data.rename(columns={'Potential Energy (kJ/mole)': 'U_total'})
    data['U_PME'] = data['Energy[0] (kJ/mole)'] - data['U_total']
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

results.to_csv(f'{solute}-electrostatic-initial.csv')

#-----------------------------------------------------------------------------------------
# REWEIGHTING

variables = dict(lbda=[], hE=[], hpE=[])
for point in range(npoints):
    lbda = point/(npoints - 1)
    variables['lbda'].append(lbda)
    variables['hE'].append(lbda)
    variables['hpE'].append(1)

properties = dict(
    W='exp(-beta*(U_PME - U_SC))',
    Up='hpE*U_SC' 
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

reweighting.to_csv(f'{solute}-electrostatic-reweighting.csv')

#-----------------------------------------------------------------------------------------
# FINAL FREE ENERGY

deltaG = reweighting['Fcorr'].iloc[-1]

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
