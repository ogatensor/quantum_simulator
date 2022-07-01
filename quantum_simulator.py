import numpy as np
import json
from collections import Counter

# A dictionary containing the standard quantum gates 
gates = {
    "i":np.identity(2),
    "h":np.array([[1/np.sqrt(2), 1/np.sqrt(2)],[1/np.sqrt(2), -1/np.sqrt(2)]]),
    "x":np.array([[0, 1],[1, 0]]),
    "y":np.array([[0, +1j],[-1j, 0]]),
    "z":np.array([[1, 0],[0, -1]]),
    "s":np.array([[1, 0],[0, np.exp(np.pi * +1j / 2)]]),
    "t":np.array([[1, 0],[0, np.exp(np.pi * +1j / 4)]]),
}

def get_ground_state(num_qubits):
    """
    Takes the number of qubits and returns a vector in the ground state representing |0>
    @param num_qubits: number of qubits in the system 
    @return: vector of 2^n, where n is num_qubits, representing state |0> 
    """
    vector = [0 for i in range(2**num_qubits)]
    vector[0] = 1
    return np.array(vector)

def get_single_qubit_operator(total_qubits, gate, target):
    """
    Takes the number of qubits and returns a matrix operator of size 2^n x 2^n, where n is total_qubits, for required gate
    @param total_qubits: count of qubits in the system
    @param gate: the input gate for which the operator is needed
    @param target: the target qubit(s) on which the gate is applied in the representation defined in [Link: shorturl.at/lpqxQ]
    @return: matrix operator of size 2^n x 2^n, where n is total_qubits
    """
    #starting tensor product with identity or the gate itself
    operator = gate if target == 0 else gates['i']

    for qubit in range(1, total_qubits):
        if qubit == target:
            #tensor product with the gate
            operator = np.kron(operator, gate)
        else:
            #tensor product with identity
            operator = np.kron(operator, gates['i'])
    
    return operator

def get_cx_operator(total_qubits, control, target):
    """
    Takes the number of qubits and returns a matrix operator of CNOT of size 2^n x 2^n, where n is total_qubits
    @param total_qubits: count of qubits in the system
    @param gate: the input gate for which the operator is needed; CNOT in this case
    @param target: the target qubit(s) in the representation defined in [Link: shorturl.at/lpqxQ]
    @return: CNOT matrix operator of size 2^n x 2^n, where n is total_qubits
    """
    P0x0 = np.array([[1, 0],[0, 0]])
    P1x1 = np.array([[0, 0],[0, 1]])
    X = gates['x']

    # case 1
    operator1 = P0x0 if control == 0 else gates['i']

    for qubit in range(1, total_qubits):
        if qubit == control:
            operator1 = np.kron(operator1, P0x0)
        else:
            operator1 = np.kron(operator1, gates['i'])

    # case 2
    operator2 = P1x1 if control == 0 else gates['i']

    for qubit in range(1, total_qubits):
        if qubit == control:
            operator2 = np.kron(operator2, P1x1)
        elif qubit == target:
            operator2 = np.kron(operator2, X)
        else:
            operator2 = np.kron(operator2, gates['i'])
    
    #take the sum of both cases to generate the final matrix operator
    return operator1 + operator2

def get_operator(total_qubits, gate_unitary, target_qubits):
    """
    Wrapper function for generating matrix operator to handle CNOT and all other gates separately
    @param total_qubits: count of qubits in the system
    @param gate_unitary: the input gate for which the operator is needed
    @param target_qubits: the qubit(s) onto which the gate is to be applied
    @return: matrix operator of size 2^n x 2^n, where n is total_qubits
    """
    # return unitary operator of size 2**n x 2**n for given gate and target qubits

    if gate_unitary == 'cx':
        return get_cx_operator(total_qubits, target_qubits[0], target_qubits[1])
    else:
        return get_single_qubit_operator(total_qubits, gates[gate_unitary] , target_qubits[0])


def run_program(initial_state, program):
    """
    Main function that builds and runs the quantum circuit
    @param initial_state: a state vector of n-qubits as input to the quantum system  
    @param program: an array of objects, each containing "gate" and "target" for defining the gate operation and target qubits for it 
    @return: final/evolved state of the quantum system after running the program
    """
    total_qubits = int(np.log2(len(initial_state)))

    state = initial_state
    
    #for each instruction:
    #   get matrix operator
    #   apply it to state
    #   move ahead
    for instruction in program: 
        gate = instruction['gate']
        targets = instruction['target']
        operator = get_operator(total_qubits, gate, targets)
        state = np.dot(operator, state)

    return state

def measure_all(state_vector):
    """
    Measures qubits by taking them from |+> and |-> basis to the computational basis 
    @param state_vector: a state vector of n-qubits that are to be measured  
    @return: index (or position) of the quantum state measured
    """
    # choose element from state_vector using weighted random and return it's index
    probabilities = np.abs(state_vector)**2
    index = np.random.choice(a=len(state_vector), p=probabilities)

    return index

def get_counts(state_vector, num_shots):
    """
    Executes quantum circuit num_shots times and return the probability distribution of each output through a dictionary 
    @param state_vector: a state vector of n-qubits
    @param num_shots: number of times that the program is executes  
    @return: dictionary containing the output state and its frequency
    """
    # simply execute measure_all in a loop num_shots times and
    # return object with statistics in following form:
    #   {
    #      element_index: number_of_ocurrences,
    #      element_index: number_of_ocurrences,
    #      ...
    #   }

    results = []
    
    num_bits = int(np.log2(len(state_vector)))
    

    for _ in range(num_shots):
        result = measure_all(state_vector)
        results.append("{0:b}".format(result).zfill(num_bits)[::-1]) 

    stats = Counter(results)   

    return json.dumps(stats, sort_keys=True, indent=4)

def get_random_program(num_qubits, num_instructions):
    """
    Generates a random program of num_instructions instructions for num_qubits qubits
    @param num_qubits: number of qubits in the system
    @param num_instructions: number of instructions in the program
    @return: a random program of num_instructions instructions for num_qubits qubits
    """
    # generate a random program of num_instructions instructions for num_qubits qubits
    # return a list of dictionaries containing "gate" and "target" for defining the gate operation and target qubits for it 
    program = []
    for _ in range(num_instructions):
        gate = random.choice(list(gates.keys()))
        target = random.choice(list(range(num_qubits)))
        program.append({'gate': gate, 'target': [target]})

    return program

def get_random_state(num_qubits):
    """
    Generates a random state vector of num_qubits qubits
    @param num_qubits: number of qubits in the system
    @return: a random state vector of num_qubits qubits
    """
    # generate a random state vector of num_qubits qubits
    # return a state vector of size 2**num_qubits
    state = np.random.rand(2**num_qubits)
    return state 

def get_random_program_and_state(num_qubits, num_instructions):
    """
    Generates a random program of num_instructions instructions for num_qubits qubits and a random state vector of num_qubits qubits
    @param num_qubits: number of qubits in the system
    @param num_instructions: number of instructions in the program
    @return: a random program of num_instructions instructions for num_qubits qubits and a random state vector of num_qubits qubits
    """
    # generate a random program of num_instructions instructions for num_qubits qubits and a random state vector of num_qubits qubits
    # return a list of dictionaries containing "gate" and "target" for defining the gate operation and target qubits for it 
    program = get_random_program(num_qubits, num_instructions)
    state = get_random_state(num_qubits)

    return program, state

def get_random_program_and_state_and_counts(num_qubits, num_instructions, num_shots):
    """
    Generates a random program of num_instructions instructions for num_qubits qubits and a random state vector of num_qubits qubits
    @param num_qubits: number of qubits in the system
    @param num_instructions: number of instructions in the program
    @param num_shots: number of times that the program is executes  
    @return: a random program of num_instructions instructions for num_qubits qubits and a random state vector of num_qubits qubits and the probability distribution of each output
    """
    # generate a random program of num_instructions instructions for num_qubits qubits and a random state vector of num_qubits qubits
    # return a list of dictionaries containing "gate" and "target" for defining the gate operation and target qubits for it 
    program, state = get_random_program_and_state(num_qubits, num_instructions)
    counts = get_counts(state, num_shots)

    return program, state, counts

def generate_time_crystal(num_qubits, num_instructions, num_shots, frequency, num_time_crystals, num_time_crystals_per_frequency, num_time_crystals_per_frequency_per_instruction):
    """
    Generates a time crystal of num_qubits qubits and num_instructions instructions for num_shots times
    @param num_qubits: number of qubits in the system
    @param num_instructions: number of instructions in the program
    @param num_shots: number of times that the program is executes  
    @return: a time crystal of num_qubits qubits and num_instructions instructions for num_shots times
    """
    # generate a time crystal of num_qubits qubits and num_instructions instructions for num_shots times
    # return a list of dictionaries containing "gate" and "target" for defining the gate operation and target qubits for it 
    program, state, counts = get_random_program_and_state_and_counts(num_qubits, num_instructions, num_shots)
    time_crystal = []
    for i in range(num_time_crystals):
        time_crystal.append({'program': program, 'state': state, 'counts': counts})

    return time_crystal

# Define program:
my_circuit = [
{ "gate": "h", "target": [0] }, 
{ "gate": "cx", "target": [0, 1] }
]

# Create "quantum computer" with 2 qubits (this is actually just a vector :) )
my_qpu = get_ground_state(2)

# Run circuit
final_state = run_program(my_qpu, my_circuit)

# Read results
counts = get_counts(final_state, 1000)
print('Result: ', counts)

