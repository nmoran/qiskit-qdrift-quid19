import numpy as np
import logging

from qiskit.aqua.algorithms.single_sample import IQPE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IQPEHack(IQPE):

    def __init__(self, operator, state_in, num_time_slices=1, num_iterations=1,
                 expansion_mode='suzuki', expansion_order=2,
                 shallow_circuit_concat=False, error=1.0):
        self.error = error
        super(IQPEHack, self).__init__(operator, state_in, num_time_slices, num_iterations,
                 expansion_mode, expansion_order,
                 shallow_circuit_concat=False)

    def _setup(self):
        super(IQPEHack, self)._setup()

        # now we override the pauli operators returned by regular setup
        # with qdrift ones
        error=self.error
        pauli_list = self._pauli_list
        logger.info(f'Number paulis: {len(pauli_list)}')
        logger.info('Pauli coefs: {}'.format([x[0] for x in pauli_list]))
        norm = np.sum(np.abs([x[0] for x in pauli_list]))
        logger.info(f'Norm: {norm}')
        probs = [np.abs(x[0])/norm for x in pauli_list]

        evo_time = -2 * np.pi # this is value hardcoded in iqpe.py
        num_steps = np.math.ceil((2 * norm * evo_time) ** 2 / error)
        logger.info(f'Number of steps for qdrift is {num_steps}')
        num_terms = len(pauli_list)
        standard_timestep = evo_time*norm/num_steps
        random_pauli_list=[]
        for i in range(num_steps):
            idx = np.random.choice(num_terms, p=probs)
            random_pauli_list.append([standard_timestep*np.sign(pauli_list[idx][0]),
                                      pauli_list[idx][1]])
        slice_pauli_list = random_pauli_list
        self._slice_pauli_list = slice_pauli_list
