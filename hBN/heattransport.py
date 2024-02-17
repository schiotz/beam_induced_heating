import ase
import nequip
from nequip.ase import NequIPCalculator
from ase import Atoms,  units
from ase.io import read,write
from ase.visualize import view
from ase.io.trajectory import Trajectory
import numpy as np
from ase.geometry import get_layers
from ase.build import surface, bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nvtberendsen import NVTBerendsen
from ase.cluster.cubic import FaceCenteredCubic
import numpy as np
import torch
from ase.io import read, write
from ase.md import VelocityVerlet, Langevin
from ase import units
import random

substrate_to_use="hbn"
nanoparticlesize="superbig"
temp = 300 #424
oxy=5

temp=450

###########

atoms=read("hbn.POSCAR")
atoms=atoms.repeat((27,27,4))
atoms.cell[2,2]+=12
atoms.center(axis=2)
substrate=atoms
zsurface  = max(substrate.positions[::,2])

#nanoparticle

surfaces = [(1, 1, 1),(-1,-1,-1),(1,0,0)] # These are the surfaces I will include
if nanoparticlesize=="small":
    layers=[4,2,5]
if nanoparticlesize=="medium":
    layers=[5,3,6]
if nanoparticlesize=="big":
    layers=[6,5,7]
if nanoparticlesize=="superbig":
    layers=[7,4,8]
    
    
np = FaceCenteredCubic('Au', surfaces, layers)
np.rotate((1,1,1),(0,0,1))
np.cell = substrate.cell
np.center()
np.translate([0,0,1])
np.rotate(v=(0,0,1),a=75,center='COU')


znp  = min(np.positions[::,2])
zsize = max(np.positions[::,2])-min(np.positions[::,2])
np.translate([0,0,-znp+zsurface+2.0])
znp  = min(np.positions[::,2])
np.translate([0,0,-znp+zsurface+2.1])
# np.rotate(v=(0,0,1),a=30,center='COU')

atoms = substrate + np
atoms.center(vacuum=5,axis=2)


t1 = Trajectory('atoms_to_run.traj', 'a')
t1.write(atoms)
atoms = read("atoms_to_run.traj")


#set calculator
calculator = NequIPCalculator.from_deployed_model(
    model_path="14poth.pth",
    species_to_type_name = {
        "Au": "Au",
        "N": "N",
        "B": "B"
    },
    # energy_units_to_eV=0.043,
    device='cuda')


del np 
import numpy as np
temp = np.zeros((len(atoms),3))
for i in atoms:
    index=i.index
    if i.symbol=="Au":
        temp[index]=[600,600,600]
    else: temp[index]=[300,300,300]




atoms.calc = calculator
atoms.pbc = [True,True,False]

trajectory = Trajectory("substrate{}_nanoparticle111{}oxygen.traj".format(substrate_to_use,nanoparticlesize), "w", atoms)

#MaxwellBoltzmannDistribution(atoms, temperature_K=1*temp) #give momenta to atoms according to 300K
#MaxwellBoltzmannDistribution(atoms, temperature_K=0.5*300) #give momenta to atoms according to 300K
dyn = Langevin(atoms, timestep=1*units.fs, temperature=temp*units.kB,friction=0.005)
dyn.attach(trajectory, interval=50)
dyn.run(1550)
del atoms[(atoms.positions[::,2]-atoms.cell[2,2]/2)**2>(atoms.cell[2,2]/2)**2]
del np
import numpy as np
temp = np.zeros((len(atoms),3))
for i in atoms:
    index=i.index
    if i.symbol=="Au":
        temp[index]=[600,600,600]
    else: temp[index]=[300,300,300]

        
for i in range(30):
    dyn = Langevin(atoms, timestep=1*units.fs, temperature=temp*units.kB,friction=0.005)
    dyn.attach(trajectory, interval=100)
    dyn.run(500)
    del atoms[(atoms.positions[::,2]-atoms.cell[2,2]/2)**2>(atoms.cell[2,2]/2)**2]
    del np
    import numpy as np
    temp = np.zeros((len(atoms),3))
    for i in atoms:
        index=i.index
        if i.symbol=="Au":
            temp[index]=[600,600,600]
        else: temp[index]=[300,300,300]
    
for i in range(200):
    dyn = VelocityVerlet(atoms, 2*units.fs)
    dyn.attach(trajectory, interval=50) 
    dyn.run(500)
    del atoms[(atoms.positions[::,2]-atoms.cell[2,2]/2)**2>(atoms.cell[2,2]/2)**2]
