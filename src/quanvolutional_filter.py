import random

import numpy as np
import qiskit

from sqrt_swap_gate import SqrtSwapGate


class QuanvolutionalFilter:
    """Quanvolutional filter class.
    """
    def __init__(self, kernel_size: tuple[int, int]):
        """Initialise the circuit.

        :param tuple[int, int] kernel_size: the kernel size.
        """
        self.num_qubits: int = kernel_size[0] * kernel_size[1]
        
        # Step 0: Build a quantum circuit as a filter.
        self.__build_initial_circuit()
        
        # Step 1: assign a connection probability between each qubit.
        self.connection_probabilities = {}
        self.__set_connection_probabilities()
        
        # Step 2: Select a two-qubit gate according to the connection probabilities.
        self.selected_gates = []
        # Define the set of two-qubits gates.
        self.two_qubit_gates = [
            qiskit.circuit.library.CXGate(),
            qiskit.circuit.library.SwapGate(),
            SqrtSwapGate(),
            qiskit.circuit.library.CUGate
        ]  # according to the paper
        # Select two-qubit gates to the circuit.
        self.__select_two_qubit_gates()
        
        # Step 3: Select one-qubit gates.
        # Define the set of one-qubit gates.
        self.one_qubit_one_parameterised_gates = [
            qiskit.circuit.library.RXGate,
            qiskit.circuit.library.RYGate,
            qiskit.circuit.library.RZGate,
            qiskit.circuit.library.PhaseGate
        ]
        self.one_qubit_three_parameterised_gates = [
            qiskit.circuit.library.UGate
        ]
        self.one_qubit_non_parametrised_gates = [
            qiskit.circuit.library.TGate(),
            qiskit.circuit.library.HGate()
        ]
        self.one_qubit_gates = (
            self.one_qubit_one_parameterised_gates + \
            self.one_qubit_three_parameterised_gates + \
            self.one_qubit_non_parametrised_gates
        )
        # Select one-qubit gates.
        self.num_one_qubit_gates = np.random.rand() * (2 * self.num_qubits**2)
        self.__select_one_qubit_gates()
        
        # Step 4: Apply the randomly selected gates at an random order.
        self.__apply_selected_gates()
        
        # Step 5: Set measurements to the lot qubits.
        self.circuit.measure_all()

    def __build_initial_circuit(self):
        """Build the initial ciruit.
        """
        self.quantum_register = qiskit.QuantumRegister(
            size=self.num_qubits,
            name="encoded_data"
        )
        self.classical_register = qiskit.ClassicalRegister(
            size=self.num_qubits,
            name="decoded_data"
        )
        self.circuit = qiskit.QuantumCircuit(
            [self.quantum_register, self.classical_register],
            name="quanvolutional_filter"
        )
        
    def __set_connection_probabilities(self):
        """Randomly assign a connection probability between each qubit.
        """
        for index in range(self.num_qubits):
            for next_index in range(index, self.num_qubits):
                # Get a random connection probability.
                connection_probability = np.random.rand()
                
                # Set the connection probability.
                self.connection_probabilities[(index, next_index)] = connection_probability
    
    def __select_two_qubit_gates(self):
        """Select two-qubit gates.
        """
        for key, value in self.connection_probabilities.items():
            if value <= 0.5:
                # Skip the pair.
                pass
            
            # Select a two-qubit gate.
            selected_gate = random.choice(self.two_qubit_gates)
            
            # Set random parameters to the CU gate.
            if selected_gate == qiskit.circuit.library.CUGate:
                cu_params = np.random.rand(4) * (2 * np.pi)
                selected_gate = qiskit.circuit.library.CUGate(
                    theta=cu_params[0],
                    phi=cu_params[1],
                    lam=cu_params[2],
                    gamma=cu_params[3]
                )
            
            # Shuffle the qubits to rnadomly decide on the target and controlled qubits.
            shuffled_key = key
            if value <= 0.75:
                shuffled_key[0] = key[1]
                shuffled_key[1] = key[0]
                
            # Keep the selected gate.
            self.selected_gates.append((selected_gate, shuffled_key))
            
    def __select_one_qubit_gates(self):
        """Select one-qubit gates.
        """
        for _ in range(self.num_one_qubit_gates):
            # Select a two-qubit gate.
            selected_gate = random.choice(self.one_qubit_gates)
            
            # Set random parameters to the CU gate.
            if selected_gate in self.one_qubit_one_parameterised_gates:
                gate_params = np.random.rand(1) * (2 * np.pi)
                selected_gate = selected_gate(gate_params[0])
            elif selected_gate in self.one_qubit_three_parameterised_gates:
                gate_params = np.random.rand(3) * (2 * np.pi)
                selected_gate = selected_gate(gate_params[0], gate_params[1], gate_params[3])
            
            # Decide the target qubit.
            target_qubit = np.random.randint(low=0, high=self.num_qubits-1)
                
            # Keep the selected gate.
            self.selected_gates.append((selected_gate, [target_qubit]))
    
    def __apply_selected_gates(self):
        """Apply the selected gates to the circuit.
        """
        # Shuffle the order of gates.
        random.shuffle(self.selected_gates)
        
        for gate, qubits in self.selected_gates:
            self.circuit.append(gate, qubits)
