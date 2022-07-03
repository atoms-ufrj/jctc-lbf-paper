import argparse
import ufedmm
import pandas as pd
from simtk import openmm, unit
from simtk.openmm import app
from ufedmm import cvlib

parser = argparse.ArgumentParser()
parser.add_argument('--solute', dest='solute', help='the solute name', default='phenol')
parser.add_argument('--lambda', dest='lbda', help='the lambda value', type=float, default=1.0)
parser.add_argument('--platform', dest='platform', help='the computation platform', default='CUDA')
args = parser.parse_args()

base = f'{args.solute}-{int(100*args.lbda):03d}'

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

start  = dict(C=0.0, D=0.4, E=0.6)
finish = dict(C=0.7, D=0.8, E=1.0)
U = '(hC-hD)*U_BG + hD*U_LJ + hE*U_SC'
for a in ['C', 'D', 'E']:
    U += f'; h{a}=(step(x{a})-step(x{a}-1))*(6*x{a}^2-15*x{a}+10)*x{a}^3+step(x{a}-1)'
    U += f'; x{a}=(lambda-{start[a]})/{finish[a]-start[a]}'
cv_force = openmm.CustomCVForce(U)
cv_force.addGlobalParameter('lambda', args.lbda)
cv_force.addCollectiveVariable('U_BG', U_BG)
cv_force.addCollectiveVariable('U_LJ', U_LJ)
cv_force.addCollectiveVariable('U_SC', U_SC)

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
    globalParameterStates=pd.DataFrame({'inOutCoulombScaling': [1.0]}),
    collectiveVariables=True
    )
    
simulation.reporters.append(data_reporter)
simulation.step(nsteps)


