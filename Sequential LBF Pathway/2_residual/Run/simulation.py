import argparse
import ufedmm
import numpy as np
import pandas as pd
from simtk import openmm, unit
from simtk.openmm import app
from ufedmm import cvlib

parser = argparse.ArgumentParser()
parser.add_argument('--solute', dest='solute', help='the solute name', default='phenol')
parser.add_argument('--state', dest='state', help='the state of lambda value', type=int, default=0)
parser.add_argument('--platform', dest='platform', help='the computation platform', default='CUDA')
args = parser.parse_args()


lambdas_vdw = np.array([0.00, 0.25, 0.50, 0.75, 1.00]) 

lbda = lambdas_vdw[args.state]

base = f'{args.solute}-vdw-decapped-{int(100*lbda):03d}'

dt = 2*unit.femtoseconds
totaltime = 24*unit.nanoseconds
barostat_interval = 50*unit.femtoseconds
sampling_interval = 200*unit.femtoseconds

temp = 298.15*unit.kelvin
pressure = 1*unit.atmosphere
rcut = 12*unit.angstroms
rswitch = 11*unit.angstroms
gamma = 10/unit.picoseconds

nsteps = round(totaltime/dt)
report_interval = round(sampling_interval/dt)
barostat_freq = round(barostat_interval/dt)

platform = openmm.Platform.getPlatformByName(args.platform)

gro = app.GromacsGroFile(f'{args.solute}-in-water.gro')
top = app.GromacsTopFile(
    f'{args.solute}-in-water.top',
    periodicBoxVectors=gro.getPeriodicBoxVectors()
    )

system = top.createSystem(
        nonbondedCutoff=rcut,
        switchDistance=rswitch,
        nonbondedMethod=app.PME,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True
        )

solute_atoms = [atom.index for atom in top.topology.atoms() if atom.residue.name == 'MOL']
nbforce = next(filter(lambda f: isinstance(f, openmm.NonbondedForce), system.getForces()))

U_LJ = cvlib.InOutLennardJonesForce(solute_atoms, nbforce) 
U_BG = U_LJ.capped_version(m=3)
U_SC = cvlib.InOutCoulombForce(solute_atoms, nbforce) 

U = 'U_BG + hD*(U_LJ-U_BG)'
U += '; hD=lambda'
cv_force = openmm.CustomCVForce(U)
cv_force.addGlobalParameter('lambda', lbda)
cv_force.addCollectiveVariable('U_BG', U_BG)
cv_force.addCollectiveVariable('U_LJ', U_LJ)
system.addForce(cv_force)

system.addForce(openmm.MonteCarloBarostat(pressure, temp, barostat_freq))
integrator = openmm.LangevinMiddleIntegrator(temp, gamma, dt)
simulation = app.Simulation(top.topology, system, integrator, platform)
simulation.context.setPositions(gro.positions)
simulation.context.setVelocitiesToTemperature(temp)

data_reporter = ufedmm.StateDataReporter(
    f'{base}.csv',
    report_interval,
    separator=',',
    step=True,
    potentialEnergy=True,
    kineticEnergy=True,
    temperature=True,
    density=True,
    collectiveVariables=True
    )

simulation.reporters.append(data_reporter)

simulation.step(nsteps)


