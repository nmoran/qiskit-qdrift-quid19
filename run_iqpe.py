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
import concurrent.futures
import multiprocessing as mp
import os
import gc

import compute_energies as ce

if __name__ == '__main__':
    # Create parser with args to control behaviour
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--processes', type=int, default=1, help='Number of processes to use (default=1)')
    parser.add_argument('-r', '--no-ref', action='store_true', help='Do not calculate reference values using exact eigensolver.')
    parser.add_argument('-n', '--no-hack', action='store_true', help='Do not attempt to use the modified IQPE.')
    parser.add_argument('-i', '--include-standard-iqpe', action='store_true', help='Include the standard IQPE method.')
    parser.add_argument('-s', '--steps', type=int, default=10, help='Number of distance steps to use between 0.5 and 1.0 (default=10).')
    parser.add_argument('-f', '--first_atom', default='H', help='The first atom (default=H).')
    parser.add_argument('-e', '--error', type=float, default=0.1, help='The error to use for qdrift IQPE (default=0.1).')
    parser.add_argument('-v', '--verbose', action='store_true')

    # parse command line args
    opts = parser.parse_args(sys.argv[1:])

    # if verbose flag set verbosity level
    if opts.verbose:
        #set_qiskit_chemistry_logging(logging.INFO)
        logging.basicConfig(level=logging.INFO)


    algorithms = []
    if not opts.no_hack:
        algorithms.append('iqpe_hack')
    if not opts.no_ref:
        algorithms.append('exacteigensolver')
    if opts.include_standard_iqpe:
        algorithms.append('iqpe')

    start = 0.5  # Start distance
    by    = 1.0  # How much to increase distance by
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
                result = ce.compute_energy(
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
                                    ce.compute_energy,
                                    i,
                                    d,
                                    algorithm,
                                    opts.first_atom,
                                    error=opts.error
                    )
                    futures_to_algorithms[future] = algorithm
            logging.info(f'Loaded {len(futures_to_algorithms)} tasks and waiting for completion')
            for future in concurrent.futures.as_completed([x for x in futures_to_algorithms]):
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


    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
