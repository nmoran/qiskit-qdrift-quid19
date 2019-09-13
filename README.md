# qiskit-qdrift-quid19
Prototype implementation of qdrift algorithm for quantum chemistry simulation from [https://arxiv.org/abs/1811.08017](https://arxiv.org/abs/1811.08017).

In order to solve the important electronic structure problem, it is import to be able to calculate the energy of a given moleucle. One of the ways to do this is by preparing a state, time evolving under the Hamiltonian for a sequence of times, and measuring the phases. From this, we can infer the energy.

The time evolution is a difficult step especialy in quantum chemistry where our Hamiltonian may have O(n^4) terms.

The standard Trotter approximations U(t) ~= V(t) = \prod_j exp(-i(t)h_j H_j) for t small.

The algorithm we have implemented here approaches the problem of decomposing U(t) in a slightly different way. 

We set a standard rotation angle for all the terms in the Hamiltonian, and then generate a sequence of rotations as detailed in the paper.

This method was reported to be able to reduce the gates required to simulate the time evolution by up to 1000x.

We have implemented Iterative Phase Estimation to measure the energy.


# Usage of our code

Requirements: Qiskit(0.12)

To get the gate counts for H2 and LiH, run the files H2_gate_counts.py and LiH_gate_counts.py or edit them for whatever molecule you choose.

To generate the plots of energy for H2, use the run_iqpe.py
