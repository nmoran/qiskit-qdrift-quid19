
import numpy as np
import matplotlib.pyplot as plt
import math

from qiskit import Aer, IBMQ, QuantumRegister, QuantumCircuit
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import noise

# lib from Qiskit Aqua
#from qiskit.aqua.operator import construct_evolution_circuit
from qiskit.aqua.operators.common import evolution_instruction
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import COBYLA, SPSA, L_BFGS_B
from qiskit.aqua.components.variational_forms import RY, RYRZ, SwapRZ


# lib from Qiskit Aqua Chemistry

from qiskit.chemistry import QiskitChemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

# setup qiskit.chemistry logging

import logging
from qiskit.chemistry import set_qiskit_chemistry_logging

driver = PySCFDriver(atom='H .0 .0 .0; Li .0 .0 1.6', unit=UnitsType.ANGSTROM,
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()

nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2
"""
print("HF energy: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy))
print("# of electrons: {}".format(num_particles))
print("# of spin orbitals: {}".format(num_spin_orbitals))
"""
# get the integrals from molecule object
h1 = molecule.one_body_integrals
h2 = molecule.two_body_integrals

# create a fermionic operator
ferOp = FermionicOperator(h1=h1, h2=h2)

qubitOp = ferOp.mapping(map_type='jordan_wigner', threshold=10**-10)
qubitOp.chop(10**-10)

num_terms = len(qubitOp.paulis)
max_term = max([np.abs(qubitOp.paulis[i][0]) for i in range(num_terms)])

error=1
evo_time = .01
#trotter_steps = math.ceil(num_terms*max_term*evo_time)**2 / 2*error
#num_time_slices = 4
num_time_slices = math.ceil((num_terms*max_term*evo_time)**2 / 2*error)
print(num_time_slices)
trotter_evo = qubitOp.evolve(evo_time = evo_time,num_time_slices = num_time_slices,expansion_mode = 'trotter')


norm = 0
probs = []
for i in range(len(qubitOp.paulis)):
    norm += np.abs(qubitOp.paulis[i][0])
for i in range(len(qubitOp.paulis)):
    probs.append(np.abs(qubitOp.paulis[i][0])/norm)
# determine the number of time steps required and the duration of the standardized timestep
num_steps = math.ceil((2*norm*evo_time)**2 /error)
standard_timestep = evo_time*norm/num_steps

random_pauli_list=[]
for i in range(num_steps):
    idx = np.random.choice(num_terms,p=probs)
    #form the list keeping track of the sign of the coefficients
    random_pauli_list.append([np.sign(qubitOp.paulis[idx][0])*standard_timestep,qubitOp.paulis[idx][1]])

instruction=evolution_instruction(random_pauli_list, evo_time=evo_time, num_time_slices=num_time_slices,
                          controlled=False, power=1,
                          use_basis_gates=True, shallow_slicing=False)

quantum_registers = QuantumRegister(qubitOp.num_qubits)
qc = QuantumCircuit(quantum_registers)
qc.append(instruction, quantum_registers)
qc = qc.decompose()
print('trotter gate counts')
print(trotter_evo.count_ops())
print('qdrift gate counts')
print(qc.count_ops())