#!/usr/bin/env python3

"""
Script to run different qiskit Quantum Phase Estimation methods for comparison

Adapted from notebook at
https://github.com/Qiskit/qiskit-community-tutorials/blob/master/chemistry/h2_iqpe.ipynb
"""
import time
import numpy as np
import multiprocessing as mp
import argparse
import sys
import logging
import concurrent.futures
import multiprocessing as mp
import os
import gc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def garbage_collected(func):
    def decorated_func(*args, **kwargs):
        result = func(*args, **kwargs)
        _ = gc.collect()
        return result
    return decorated_func

@garbage_collected
def compute_energy(i, distance, algorithm, first_atom='H', sim='statevector_simulator', error=0.1):
    logging.info(f"Running distance {i} with algorithm {algorithm}...")
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
                            shallow_circuit_concat=True, error=error)
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


if __name__ == '__main__':
    # Create parser with args to control behaviour
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--processes', type=int, default=1, help='Number of processes to use (default=1)')
    parser.add_argument('-r', '--no-ref', action='store_true', help='Do not calculate reference values using exact eigensolver.')
    parser.add_argument('-i', '--include-standard-iqpe', action='store_true', help='Include the standard IQPE method.')
    parser.add_argument('-s', '--steps', type=int, default=10, help='Number of distance steps to use between 0.5 and 1.0 (default=10).')
    parser.add_argument('-f', '--first_atom', default='H', help='The first atom (default=H).')
    parser.add_argument('-e', '--error', default=0.1, help='The error to use for qdrift IQPE (default=0.1).')
    parser.add_argument('-v', '--verbose', action='store_true')

    # parse command line args
    opts = parser.parse_args(sys.argv[1:])

    # if verbose flag set verbosity level
    if opts.verbose:
        set_qiskit_chemistry_logging(logging.INFO)
        logging.basicConfig(level=logging.INFO)


    algorithms = ['iqpe_hack']
    if not opts.no_ref:
        algorithms.append('exacteigensolver')
    if opts.include_standard_iqpe:
        algorithms.append('iqpe')

    start = 0.5  # Start distance
    by    = 0.5  # How much to increase distance by
    steps = opts.steps   # Number of steps to increase by
    energies = {}
    energy_stds = {}
    hf_energies = np.empty(steps)
    distances = np.empty(steps)

    logging.info(f'Running for algorithms {algorithms} and {steps} steps...')

    start_time = time.time()
    if opts.processes == 1:
        for j in range(len(algorithms)):
            algorithm = algorithms[j]
            energies[algorithm] = np.empty(steps)
            energy_stds[algorithm] = np.empty(steps)
            for i in range(steps):
                d = start + i*by/steps
                result = compute_energy(
                    i,
                    d,
                    algorithm,
                    opts.first_atom,
                    error=opts.error
                )
                i, d, energy, hf_energy, energy_error = result
                energies[algorithm][i] = energy
                energy_stds[algorithm][i] = energy_error
                hf_energies[i] = hf_energy
                distances[i] = d
    else:
        max_workers = opts.processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures_to_algorithms = {}
            for j in range(len(algorithms)):
                algorithm = algorithms[j]
                energies[algorithm] = np.empty(steps)
                energy_stds[algorithm] = np.empty(steps)
                for i in range(steps):
                    d = start + i*by/steps
                    future = executor.submit(
                                    compute_energy,
                                    i,
                                    d,
                                    algorithm,
                                    opts.first_atom,
                                    error=opts.error
                    )
                    futures_to_algorithms[future] = algorithm
            logging.info(f'Loaded {len(futures_to_algorithms)} tasks and waiting for completion')
            for future in concurrent.futures.as_completed(futures_to_algorithms):
                i, d, energy, hf_energy, energy_error = future.result()
                algorithm = futures_to_algorithms[future]
                energies[algorithm][i] = energy
                energy_stds[algorithm][i] = energy_error
                hf_energies[i] = hf_energy
                distances[i] = d

    print(' --- complete')

    print('Distances: ', distances)
    print('Energies:', energies)
    print('Energy Stds:', energy_stds)
    print('Hartree-Fock energies:', hf_energies)

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.plot(distances, hf_energies, label='Hartree-Fock', alpha=0.5, marker='+')
    for algorithm, es in energies.items():
        plt.errorbar(distances, es, yerr=energy_stds[algorithm], label=algorithm, alpha=0.5, marker='+')
    plt.xlabel('Interatomic distance')
    plt.ylabel('Energy')
    plt.title(f'{opts.first_atom}-H Ground State Energy')
    plt.legend(loc='upper right')
    filename = 'energies_0.png'
    i = 0
    while os.path.exists(f'energies_{i}.png'): i += 1
    plt.savefig(f'energies_{i}.png')

    # we plot energy difference with reference energy if present
    if 'exacteigensolver' in energies:
        plt.figure()
        plt.plot(distances, hf_energies - energies['exacteigensolver'], label='Hartree-Fock', alpha=0.5, marker='+')
        for algorithm, es in energies.items():
            if algorithm != 'exacteigensolver':
                plt.plot(distances, es - energies['exacteigensolver'], label=algorithm, alpha=0.5, marker='+')
        plt.xlabel('Interatomic distance')
        plt.ylabel('Energy - Energy ref')
        plt.title(f'{opts.first_atom}-H Ground State Energy')
        plt.legend(loc='upper right')
        filename = 'energy_diffs_0.png'
        i = 0
        while os.path.exists(f'energy_diffs_{i}.png'): i += 1
        plt.savefig(f'energy_diffs_{i}.png')
