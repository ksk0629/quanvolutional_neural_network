import random

import numpy as np
import qiskit
import qiskit_aer

from sqrt_swap_gate import SqrtSwapGate
import utils_qnn


class QuanvFilter:
    """Quanvolutional filter class."""

    def __init__(self, kernel_size: tuple[int, int]):
        """Prepare the circuit.

        :param tuple[int, int] kernel_size: kernel size.
        """
        # Get the number of qubits.
        self.num_qubits: int = kernel_size[0] * kernel_size[1]

        # Step 0: Build a quantum circuit as a filter.
        self.__build_initial_circuit()

        # Step 1: assign a connection probability between each qubit.
        self.connection_probabilities = {}
        self.__set_connection_probabilities()

        # Step 2: Select a two-qubit gate according to the connection probabilities.
        self.selected_gates = []
        # Define the set of two-qubits gates.
        self.two_qubit_four_parameterised_gates = [qiskit.circuit.library.CUGate]
        sqrt_swap_gate = SqrtSwapGate()
        self.two_qubit_non_parameterised_gates = [
            qiskit.circuit.library.CXGate(),
            qiskit.circuit.library.SwapGate(),
            sqrt_swap_gate.get_gate(),
        ]
        self.two_qubit_gates = (
            self.two_qubit_four_parameterised_gates
            + self.two_qubit_non_parameterised_gates
        )
        # Select two-qubit gates to the circuit.
        self.__select_two_qubit_gates()

        # Step 3: Select one-qubit gates.
        # Define the set of one-qubit gates.
        self.one_qubit_one_parameterised_gates = [
            qiskit.circuit.library.RXGate,
            qiskit.circuit.library.RYGate,
            qiskit.circuit.library.RZGate,
            qiskit.circuit.library.PhaseGate,
        ]
        self.one_qubit_three_parameterised_gates = [qiskit.circuit.library.UGate]
        self.one_qubit_non_parametrised_gates = [
            qiskit.circuit.library.TGate(),
            qiskit.circuit.library.HGate(),
        ]
        self.one_qubit_gates = (
            self.one_qubit_one_parameterised_gates
            + self.one_qubit_three_parameterised_gates
            + self.one_qubit_non_parametrised_gates
        )
        # Select one-qubit gates.
        self.num_one_qubit_gates = np.random.randint(
            low=0, high=2 * self.num_qubits**2 - 1
        )
        self.__select_one_qubit_gates()

        # Step 4: Apply the randomly selected gates at an random order.
        self.__apply_selected_gates()

        # Step 5: Set measurements to the lot qubits.
        self.circuit.measure(self.quantum_register, self.classical_register)

        # Transpile the circuit.
        self.simulator = qiskit_aer.AerSimulator()

    def __build_initial_circuit(self):
        """Build the initial ciruit."""
        self.quantum_register = qiskit.QuantumRegister(
            size=self.num_qubits, name="quantum_register"
        )
        self.classical_register = qiskit.ClassicalRegister(
            size=self.num_qubits, name="classical_register"
        )
        self.circuit = qiskit.QuantumCircuit(
            self.quantum_register, self.classical_register, name="quanvolutional_filter"
        )

    def __set_connection_probabilities(self):
        """Randomly assign a connection probability between each qubit."""
        for index in range(self.num_qubits):
            for next_index in range(index + 1, self.num_qubits):
                # Get a random connection probability.
                connection_probability = np.random.rand()

                # Set the connection probability.
                self.connection_probabilities[(index, next_index)] = (
                    connection_probability
                )

    def __select_two_qubit_gates(self):
        """Select two-qubit gates."""
        for key, value in self.connection_probabilities.items():
            if value <= 0.5:
                # Skip the pair.
                pass

            # Select a two-qubit gate.
            selected_gate = random.choice(self.two_qubit_gates)

            # Set random parameters to the CU gate.
            if selected_gate in self.two_qubit_four_parameterised_gates:
                four_params = np.random.rand(4) * (2 * np.pi)
                selected_gate = qiskit.circuit.library.CUGate(
                    theta=four_params[0],
                    phi=four_params[1],
                    lam=four_params[2],
                    gamma=four_params[3],
                )

            # Shuffle the qubits to rnadomly decide on the target and controlled qubits.
            shuffled_key = [*key]  # key is tuple. Need to cast to list.
            if value <= 0.75:
                shuffled_key[0] = key[1]
                shuffled_key[1] = key[0]

            # Keep the selected gate.
            self.selected_gates.append((selected_gate, shuffled_key))

    def __select_one_qubit_gates(self):
        """Select one-qubit gates."""
        for _ in range(self.num_one_qubit_gates):
            # Select a two-qubit gate.
            selected_gate = random.choice(self.one_qubit_gates)

            # Set random parameters to the CU gate.
            if selected_gate in self.one_qubit_one_parameterised_gates:
                gate_one_param = np.random.rand(1) * (2 * np.pi)
                selected_gate = selected_gate(gate_one_param[0])
            elif selected_gate in self.one_qubit_three_parameterised_gates:
                gate_three_params = np.random.rand(3) * (2 * np.pi)
                selected_gate = selected_gate(
                    gate_three_params[0], gate_three_params[1], gate_three_params[2]
                )

            # Decide the target qubit.
            target_qubit = np.random.randint(low=0, high=self.num_qubits - 1)

            # Keep the selected gate.
            self.selected_gates.append((selected_gate, [target_qubit]))

    def __apply_selected_gates(self):
        """Apply the selected gates to the circuit."""
        # Shuffle the order of gates.
        random.shuffle(self.selected_gates)

        for gate, qubits in self.selected_gates:
            self.circuit.append(gate, qubits)

    def __load_data(self, encoded_data: np.ndarray):
        """Load the encoded data to the cirucit.

        :param np.ndarray encoded_data: encoded data
        """
        # Build the initialising part.
        initialising_part = qiskit.QuantumCircuit(
            self.quantum_register, self.classical_register
        )
        initialising_part.initialize(encoded_data)
        # Make a new circuit by composing the initialising part and self.circuit.
        self.circuit = initialising_part.compose(self.circuit)

    def draw(self):
        """Draw the circuit."""
        try:
            return self.circuit.draw(output="mpl")
        except:
            print(self.circuit.draw())

    def run(self, data: np.ndarray, shots: int) -> int:
        """Run this filter.

        :param np.ndarray data: input data, which is not encoded
        :param int shots: number of shots
        :return int: decoded result data
        """
        # Encode the data.
        encoded_data = utils_qnn.encode_with_threshold(data)

        # Load the data to the circuit.
        self.__load_data(encoded_data)

        # Run the circuit.
        transpiled_circuit = qiskit.transpile(self.circuit, self.simulator)
        result = self.simulator.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts(transpiled_circuit)

        # Decode the result data.
        decoded_data = utils_qnn.decode_by_summing_ones(counts)

        return decoded_data
