import itertools
import os
import pickle

import random

import numpy as np
import qiskit
from qiskit import qpy
import qiskit_aer

from sqrt_swap_gate import SqrtSwapGate
import utils_qnn


class QuanvFilter:
    """Quanvolutional filter class."""

    def __init__(self, kernel_size: tuple[int, int]):
        """Prepare the circuit.

        :param tuple[int, int] kernel_size: kernel size.
        """
        # Initialise the look-up table.
        self.lookup_table = None
        # Set the simulator.
        self.simulator = qiskit_aer.AerSimulator()

        # Get the number of qubits.
        self.num_qubits: int = kernel_size[0] * kernel_size[1]

        # Step 0: Build a quantum circuit as a filter.
        self.__build_initial_circuit()

        # Step 1: assign a connection probability between each qubit.
        self.connection_probabilities = {}
        self.__set_connection_probabilities()

        # Step 2: Select a two-qubit gate according to the connection probabilities.
        self.selected_gates = []
        self.__set_two_qubit_gate_set()
        self.__select_two_qubit_gates()

        # Step 3: Select one-qubit gates.
        self.__set_one_qubit_gate_set()
        self.num_one_qubit_gates = np.random.randint(
            low=0, high=2 * self.num_qubits**2 - 1
        )
        self.__select_one_qubit_gates()

        # Step 4: Apply the randomly selected gates in an random order.
        self.__apply_selected_gates()

        # Step 5: Set measurements to the lot qubits.
        self.circuit.measure(self.quantum_register, self.classical_register)

    def __build_initial_circuit(self):
        """Build the initial circuit."""
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
        """Randomly assign connection probabilities between each qubit."""
        for index in range(self.num_qubits):
            for next_index in range(index + 1, self.num_qubits):
                # Get a random connection probability.
                connection_probability = np.random.rand()

                # Set the connection probability.
                self.connection_probabilities[(index, next_index)] = (
                    connection_probability
                )

    def __set_two_qubit_gate_set(self):
        """Set the member variables related to two-qubit gates."""
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

    def __select_two_qubit_gates(self):
        """Select two-qubit gates."""
        for qubit_pair, connection_probability in self.connection_probabilities.items():
            if connection_probability <= 0.5:
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

            # Shuffle the pair of qubits to randomly decide on the target and controlled qubits.
            shuffled_qubit_pair = [*qubit_pair]  # key is tuple. Need to cast to list.
            if connection_probability <= 0.75:
                shuffled_qubit_pair[0] = qubit_pair[1]
                shuffled_qubit_pair[1] = qubit_pair[0]

            # Keep the selected gate.
            self.selected_gates.append((selected_gate, shuffled_qubit_pair))

    def __set_one_qubit_gate_set(self):
        """Set one-qubit gate set to self.one_qubit_gates variable."""
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

    def load_data(self, encoded_data: np.ndarray) -> qiskit.QuantumCircuit:
        """Load the encoded data to the circuit.

        :param np.ndarray encoded_data: encoded data
        :return qiskit.QuantumCircuit: circuit having data encoded part
        """
        # Build the initialising part.
        initialising_part = qiskit.QuantumCircuit(
            self.quantum_register, self.classical_register
        )
        initialising_part.initialize(encoded_data)
        # Make a new circuit by composing the initialising part and self.circuit.
        ready_circuit = initialising_part.compose(self.circuit)
        return ready_circuit

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
        ready_circuit = self.load_data(encoded_data)

        # Run the circuit.
        transpiled_circuit = qiskit.transpile(ready_circuit, self.simulator)
        result = self.simulator.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts(transpiled_circuit)

        # Decode the result data.
        decoded_data = utils_qnn.decode_by_summing_ones(counts)

        return decoded_data

    def make_lookup_table(self, shots: int, threshold: float = 0):
        """Make the look-up table.

        :param int shots: number of shots
        :param float threshold: threshold to encode, defaults to 0
        """
        if self.lookup_table is None:
            possible_inputs = list(
                itertools.product([threshold + 1, threshold], repeat=self.num_qubits)
            )
            vectorised_run = np.vectorize(self.run, signature="(n),()->()")
            possible_outputs = vectorised_run(np.array(possible_inputs), shots)
            self.lookup_table = {
                inputs: outputs
                for inputs, outputs in zip(possible_inputs, possible_outputs)
            }

    def get_circuit_filename(self, filename_prefix: str):
        """Get a circuit filename to save and load the circuit.

        :param str filename_prefix: prefix of filename
        """
        return f"{filename_prefix}_quanv_filter.qpy"

    def get_lookup_table_filename(self, filename_prefix: str):
        """Get a look-up table filename to save and load the circuit.

        :param str filename_prefix: prefix of filename
        """
        return f"{filename_prefix}_quanv_filter_lookup_table.pickle"

    def save(self, output_dir: str, filename_prefix: str):
        """Save the QuanvFilter.

        :param str output_dir: path to output dir
        :param str filename_prefix: prefix of output files
        """
        # Create the output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the circuit.
        circuit_filename = self.get_circuit_filename(filename_prefix=filename_prefix)
        circuit_path = os.path.join(output_dir, circuit_filename)
        with open(circuit_path, "wb") as file:
            qpy.dump(self.circuit, file)

        # Save the look-up table if it is not None.
        if self.lookup_table is not None:
            lookup_table_filename = self.get_lookup_table_filename(
                filename_prefix=filename_prefix
            )
            lookup_table_path = os.path.join(output_dir, lookup_table_filename)
            with open(lookup_table_path, "wb") as lookup_table_file:
                pickle.dump(
                    self.lookup_table,
                    lookup_table_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def load(self, input_dir: str, filename_prefix):
        """Load the quantum circuit.

        :param str input_dir: path to input dir
        :param str filename_prefix: prefix of input files
        """
        # Load the quantum circuit.
        circuit_filename = self.get_circuit_filename(filename_prefix=filename_prefix)
        circuit_path = os.path.join(input_dir, circuit_filename)
        with open(circuit_path, "rb") as circuit_file:
            # Use the first circuit as this class assumes the saved circuit file includes only one circuit data either way.
            self.circuit = qpy.load(circuit_file)[0]
        # Register the quantum register and classical registers, again here we use the first registers.
        self.quantum_register = self.circuit.qregs[0]
        self.classical_register = self.circuit.cregs[0]
        # Reset the number of qubits.
        self.num_qubits = len(self.quantum_register)

        # Load the look-up table.
        lookup_table_filename = self.get_lookup_table_filename(
            filename_prefix=filename_prefix
        )
        lookup_table_path = os.path.join(input_dir, lookup_table_filename)
        with open(lookup_table_path, "rb") as lookup_table_file:
            self.lookup_table = pickle.load(lookup_table_file)
