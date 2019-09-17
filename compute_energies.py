#!/usr/bin/env python3

"""
Script to run different qiskit Quantum Phase Estimation methods for comparison

Adapted from notebook at
https://github.com/Qiskit/qiskit-community-tutorials/blob/master/chemistry/h2_iqpe.ipynb
"""
import time
import numpy as np
import argparse
import sys
import logging
import os

def compute_energy(i, distance, algorithm, first_atom='H', sim='statevector_simulator', error=0.1):
    """
    Compute the ground state energy given a distance, method and params
    """
    from qiskit import BasicAer
    from qiskit.aqua import QuantumInstance, AquaError
    from qiskit.aqua.operators import Z2Symmetries
    from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator
    from qiskit.aqua.algorithms.single_sample import IQPE
    from qiskit.aqua.algorithms.single_sample import QPE
    from qiskit.aqua.components.iqfts import Standard
    from qiskit.aqua.algorithms.classical import ExactEigensolver
    from qiskit.chemistry import FermionicOperator
    from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
    from qiskit.chemistry.drivers import PySCFDriver, UnitsType
    from qiskit.chemistry import set_qiskit_chemistry_logging

    from IQPEHack import IQPEHack

    try:
        driver = PySCFDriver(
            atom='{} .0 .0 .0; H .0 .0 {}'.format(first_atom, distance),
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis='sto3g'
        )
    except:
        raise AquaError('PYSCF driver does not appear to be installed')
    molecule = driver.run()
    qubit_mapping = 'parity'
    fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    qubit_op = Z2Symmetries.two_qubit_reduction(to_weighted_pauli_operator(fer_op.mapping(map_type=qubit_mapping, threshold=1e-10)), 2)
    energy_std = 0.0

    if algorithm.lower() == 'exacteigensolver':
        exact_eigensolver = ExactEigensolver(qubit_op, k=1)
        result = exact_eigensolver.run()
        reference_energy = result['energy']
        energy = result['energy']
    elif algorithm.lower() == 'iqpe':
        num_particles = molecule.num_alpha + molecule.num_beta
        two_qubit_reduction = True
        num_orbitals = qubit_op.num_qubits + (2 if two_qubit_reduction else 0)

        num_time_slices = 2
        num_iterations = 8
        state_in = HartreeFock(qubit_op.num_qubits, num_orbitals,
                               num_particles, qubit_mapping, two_qubit_reduction)
        iqpe = IQPE(qubit_op, state_in, num_time_slices, num_iterations,
                    expansion_mode='trotter', expansion_order=1,
                    shallow_circuit_concat=True)
        backend = BasicAer.get_backend(sim)
        quantum_instance = QuantumInstance(backend)

        result = iqpe.run(quantum_instance)
        energy = result['energy']
    elif algorithm.lower() == 'iqpe_hack':
        num_particles = molecule.num_alpha + molecule.num_beta
        two_qubit_reduction = True
        num_orbitals = qubit_op.num_qubits + (2 if two_qubit_reduction else 0)

        num_time_slices = 1
        num_iterations = 5
        num_runs = 20
        energy_samples = np.empty(num_runs)
        for runs in range(num_runs):
            state_in = HartreeFock(qubit_op.num_qubits, num_orbitals,
                                num_particles, qubit_mapping, two_qubit_reduction)
            iqpe = IQPEHack(qubit_op, state_in, num_time_slices, num_iterations,
                            expansion_mode='trotter', expansion_order=1,
                            shallow_circuit_concat=False, error=error)
            backend = BasicAer.get_backend(sim)
            quantum_instance = QuantumInstance(backend)
            result = iqpe.run(quantum_instance)
            energy_samples[runs] = result['energy']
        energy = np.mean(energy_samples)
        energy_std = np.std(energy_samples)
    elif algorithm.lower() == 'qpe':
        num_particles = molecule.num_alpha + molecule.num_beta
        two_qubit_reduction = True
        num_orbitals = qubit_op.num_qubits + (2 if two_qubit_reduction else 0)

        num_time_slices = 10
        iqft = Standard(qubit_op.num_qubits)
        state_in = HartreeFock(qubit_op.num_qubits, num_orbitals,
                               num_particles, qubit_mapping, two_qubit_reduction)
        qpe = QPE(qubit_op, state_in, iqft, num_time_slices, num_ancillae=4,
                   expansion_mode='trotter', expansion_order=1,
                   shallow_circuit_concat=True)
        backend = BasicAer.get_backend(sim)
        quantum_instance = QuantumInstance(backend)

        result = qpe.run(quantum_instance)
        energy = result['energy']
    else:
        raise AquaError('Unrecognized algorithm.')
    return i, distance, energy + molecule.nuclear_repulsion_energy, molecule.hf_energy, energy_std


