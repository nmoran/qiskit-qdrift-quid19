import numpy as np
import matplotlib.pyplot as plt
import math

from qiskit import Aer, IBMQ, QuantumRegister, QuantumCircuit
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import noise

# lib from Qiskit Aqua
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

driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 1.6', unit=UnitsType.ANGSTROM,
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()

nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2

h1 = molecule.one_body_integrals
h2 = molecule.two_body_integrals
ferOp = FermionicOperator(h1=h1, h2=h2)
map_type = 'bravyi_kitaev'
qubitOp = ferOp.mapping(map_type=map_type, threshold=10**-10)
qubitOp.chop(10**-10)

num_terms = len(qubitOp.paulis)
max_term = max([np.abs(qubitOp.paulis[i][0]) for i in range(num_terms)])

error=.01

norm = 0
probs = []
for i in range(len(qubitOp.paulis)):
    norm += np.abs(qubitOp.paulis[i][0])
for i in range(len(qubitOp.paulis)):
    probs.append(np.abs(qubitOp.paulis[i][0])/norm)

runs = 10
print('start of big loop')
times = np.linspace(.1,1,10)
qdrift_av_counts=[]
trotter_counts=[]
#iterate through the list of durations
for time_idx in range(len(times)):
    qdrift_gate_counts = []
    num_time_slices = math.ceil((num_terms*max_term*times[time_idx])**2 / 2*error)
    #Iterate (runs) numbers of time to get average data
    for run in range(runs):
        random_pauli_list=[]
        #the number of steps from the norm, time, and error
        num_steps = math.ceil((2*norm*times[time_idx])**2 /error)
        standard_timestep = times[time_idx]*norm/num_steps
        for i in range(num_steps):
            idx = np.random.choice(num_terms,p=probs)
            #form the list keeping track of the sign of the coefficients
            random_pauli_list.append([np.sign(qubitOp.paulis[idx][0])*standard_timestep,qubitOp.paulis[idx][1]])
        instruction_qdrift=evolution_instruction(random_pauli_list, evo_time=1, num_time_slices=1, controlled=False, power=1, use_basis_gates=True, shallow_slicing=False)
        print('completed {},{} qdrift evolution_instructions'.format(str(time_idx),str(run)))
        quantum_registers_qdrift = QuantumRegister(qubitOp.num_qubits)
        qc_qdrift = QuantumCircuit(quantum_registers_qdrift)
        qc_qdrift.append(instruction_qdrift, quantum_registers_qdrift)
        qc_qdrift = qc_qdrift.decompose()
        total_qdrift = 0
        try:
            total_qdrift+=qc_qdrift.count_ops()['cx']
        except:
	        pass
        try:
            total_qdrift+=qc_qdrift.count_ops()['u1']
        except:
	        pass
        try:
            total_qdrift+=qc_qdrift.count_ops()['u2']
        except:
	        pass
        try:
            total_qdrift+=qc_qdrift.count_ops()['u3']
        except:
            pass
        qdrift_gate_counts.append(total_qdrift)
    print('start of trotter evolution instruction')
    instruction_trotter=evolution_instruction(qubitOp.paulis, evo_time=times[time_idx], num_time_slices=num_time_slices, controlled=False, power=1, use_basis_gates=True, shallow_slicing=False)
    print('end of trotter evolution instruction - on to circuit construction')
    quantum_registers_trotter = QuantumRegister(qubitOp.num_qubits)
    qc_trotter = QuantumCircuit(quantum_registers_trotter)
    qc_trotter.append(instruction_trotter, quantum_registers_trotter)
    qc_trotter = qc_trotter.decompose()
    total_trotter = 0
    try:
	    total_trotter+=qc_trotter.count_ops()['cx']
    except:
	    pass
    try:
        total_trotter+=qc_trotter.count_ops()['u1']
    except:
	    pass
    try:
	    total_trotter+=qc_trotter.count_ops()['u2']
    except:
	    pass
    try:
        total_trotter+=qc_trotter.count_ops()['u3']
    except:
	    pass
    trotter_counts.append(total_trotter)
    qdrift_av_counts.append(sum(qdrift_gate_counts)/len(qdrift_gate_counts))
    print('got through {} iterations'.format(str(time_idx)))

plt.plot(times,qdrift_av_counts,label='qdrift_avg_counts')
plt.plot(times,trotter_counts,label = 'trotter_counts')
plt.title('Gates v Duration for Time Evol({})'.format(map_type))
plt.xlabel("Duration of evolution")
plt.ylabel("Number of Gates")
plt.legend(loc=0)
plt.yscale('log')
plt.savefig("H2_gates_v_time_{}.png".format(map_type), dpi=600)
