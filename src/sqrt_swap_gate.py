import qiskit


class SqrtSwapGate():
    """Square root swap gate class.
    """
    
    def __init__(self):
        """Build the square root swap gate.
        """
        self.__qc = qiskit.QuantumCircuit(2, name='sqrt_swap')
    
        self.__qc.sx(1)
        self.__qc.cx(1, 0)
        self.__qc.ry(np.pi/4, 1)
        self.__qc.cx(0, 1)
        self.__qc.ry(-np.pi/4, 1)
        self.__qc.cx(1, 0)
        self.__qc.sx(0)

    def get_gate(self) -> qiskit.circuit.Gate:
        """Return the square root swap gate.

        :return qiskit.circuit.Gate: the square root swap gate.
        """
        return self.__qc.to_gate()