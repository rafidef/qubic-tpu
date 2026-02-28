"""
Qubic Score Addition Algorithm — Python/JAX Port
Port of score_addition.h from the Qiner reference miner.

This implements the neuroevolution-based Proof of Work for Qubic,
where mining = evolving a small ANN to learn the addition function.

The key TPU optimization: inferANN() evaluates 16,384 training pairs
independently — we batch these with jax.vmap for massive parallelism.
"""
import copy
import struct
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from qubic_k12 import kangaroo_twelve, random2


# ============================================================
# Constants (matching score_addition.h)
# ============================================================
NUMBER_OF_INPUT_NEURONS = 14       # K: 7 bits for A + 7 bits for B
NUMBER_OF_OUTPUT_NEURONS = 8       # L: 8 bits for result C
NUMBER_OF_TICKS = 120              # N: max ticks per inference
MAX_NEIGHBOR_NEURONS = 728         # 2M: each neuron has up to 728 neighbors
NUMBER_OF_MUTATIONS = 100          # S: mutation steps
POPULATION_THRESHOLD = (NUMBER_OF_INPUT_NEURONS + NUMBER_OF_OUTPUT_NEURONS +
                        NUMBER_OF_MUTATIONS)  # P = 122
TRAINING_SET_SIZE = 1 << NUMBER_OF_INPUT_NEURONS  # 2^14 = 16,384
MAX_SCORE = TRAINING_SET_SIZE * NUMBER_OF_OUTPUT_NEURONS  # 131,072
SOLUTION_THRESHOLD = MAX_SCORE * 4 // 5  # 104,857

NUMBER_OF_NEURONS = NUMBER_OF_INPUT_NEURONS + NUMBER_OF_OUTPUT_NEURONS  # 22
MAX_NUMBER_OF_NEURONS = POPULATION_THRESHOLD  # 122
MAX_NUMBER_OF_SYNAPSES = MAX_NUMBER_OF_NEURONS * MAX_NEIGHBOR_NEURONS  # 88,816
HALF_MAX_NEIGHBORS = MAX_NEIGHBOR_NEURONS // 2  # 364

# Padding for random2 output alignment
PADDING_NUMBER_OF_SYNAPSES = ((MAX_NUMBER_OF_SYNAPSES + 31) // 32) * 32

# Neuron types
NEURON_INPUT = 0
NEURON_OUTPUT = 1
NEURON_EVOLUTION = 2


# ============================================================
# Helper functions (matching score_common.h)
# ============================================================
def clamp_neuron(value: int) -> int:
    """Clamp neuron value to [-1, 1]."""
    if value > 1:
        return 1
    if value < -1:
        return -1
    return value


def to_ternary_bits(value: int, bit_count: int) -> list:
    """
    Convert integer to ternary bits.
    0 bits become -1, 1 bits stay 1.
    """
    bits = []
    for i in range(bit_count):
        bit_val = (value >> i) & 1
        bits.append(-1 if bit_val == 0 else 1)
    return bits


# ============================================================
# Training pair generation
# ============================================================
def generate_training_set():
    """Generate all 2^K possible (A, B, C=A+B) pairs."""
    half_k = NUMBER_OF_INPUT_NEURONS // 2  # 7
    bound = (1 << half_k) // 2  # 64

    training_set = []
    for A in range(-bound, bound):
        for B in range(-bound, bound):
            C = A + B
            inp = to_ternary_bits(A, half_k) + to_ternary_bits(B, half_k)
            out = to_ternary_bits(C, NUMBER_OF_OUTPUT_NEURONS)
            training_set.append((inp, out))

    return training_set


# ============================================================
# ANN Data Structures
# ============================================================
class Synapse:
    __slots__ = ['weight']
    def __init__(self, weight=0):
        self.weight = weight


class Neuron:
    __slots__ = ['type', 'value', 'mark_for_removal']
    def __init__(self, ntype=NEURON_INPUT, value=0):
        self.type = ntype
        self.value = value
        self.mark_for_removal = False


class ANN:
    """Artificial Neural Network with ring topology."""
    __slots__ = ['neurons', 'synapses', 'population']

    def __init__(self):
        self.neurons = [Neuron() for _ in range(MAX_NUMBER_OF_NEURONS)]
        # Flat array: synapse[neuron_idx * MAX_NEIGHBOR_NEURONS + neighbor_offset]
        self.synapses = [0] * MAX_NUMBER_OF_SYNAPSES
        self.population = 0

    def copy_from(self, other):
        """Deep copy another ANN's state into this one."""
        for i in range(MAX_NUMBER_OF_NEURONS):
            self.neurons[i].type = other.neurons[i].type
            self.neurons[i].value = other.neurons[i].value
            self.neurons[i].mark_for_removal = other.neurons[i].mark_for_removal
        self.synapses[:] = other.synapses[:]
        self.population = other.population


# ============================================================
# InitValue structure (matches C++ InitValue struct)
# ============================================================
class InitValue:
    """Pre-computed random values for ANN initialization."""
    def __init__(self):
        self.output_neuron_positions = [0] * NUMBER_OF_OUTPUT_NEURONS
        self.synapse_weight = [0] * (PADDING_NUMBER_OF_SYNAPSES // 32)
        self.synapse_mutation = [0] * NUMBER_OF_MUTATIONS


def parse_init_value(raw_bytes: bytes) -> InitValue:
    """Parse InitValue from raw random2 output bytes."""
    iv = InitValue()
    offset = 0

    # output_neuron_positions: numberOfOutputNeurons uint64s
    for i in range(NUMBER_OF_OUTPUT_NEURONS):
        iv.output_neuron_positions[i] = struct.unpack_from('<Q', raw_bytes, offset)[0]
        offset += 8

    # synapse_weight: paddingNumberOfSynapses/32 uint64s
    num_weight_entries = PADDING_NUMBER_OF_SYNAPSES // 32
    for i in range(num_weight_entries):
        iv.synapse_weight[i] = struct.unpack_from('<Q', raw_bytes, offset)[0]
        offset += 8

    # synapse_mutation: numberOfMutations uint64s
    for i in range(NUMBER_OF_MUTATIONS):
        iv.synapse_mutation[i] = struct.unpack_from('<Q', raw_bytes, offset)[0]
        offset += 8

    return iv


# ============================================================
# Miner class — port of score_addition::Miner
# ============================================================
class AdditionMiner:
    """
    Port of score_addition::Miner<...> from score_addition.h.
    Implements the complete Qubic addition PoW algorithm.
    """

    def __init__(self, use_jax: bool = None):
        self.current_ann = ANN()
        self.best_ann = ANN()
        self.init_value = InitValue()
        self.neuron_value_buffer = [0] * MAX_NUMBER_OF_NEURONS
        self.previous_neuron_value = [0] * MAX_NUMBER_OF_NEURONS
        self.output_neuron_expected_value = [0] * NUMBER_OF_OUTPUT_NEURONS
        self.training_set = []
        self.pool = None
        # Auto-detect JAX if not specified
        self.use_jax = HAS_JAX if use_jax is None else use_jax
        # Pre-computed numpy arrays for training data (built once)
        self._training_inputs = None
        self._training_outputs = None

    def initialize(self, mining_seed: bytes, pool: bytes):
        """Initialize with mining seed and pre-computed pool."""
        self.pool = pool
        self.training_set = generate_training_set()
        # Pre-compute training data as numpy arrays for JAX path
        if self.use_jax:
            self._precompute_training_arrays()

    def _precompute_training_arrays(self):
        """Pre-compute training input/output arrays (done once, reused every inference)."""
        inputs = np.zeros((TRAINING_SET_SIZE, NUMBER_OF_INPUT_NEURONS), dtype=np.int8)
        outputs = np.zeros((TRAINING_SET_SIZE, NUMBER_OF_OUTPUT_NEURONS), dtype=np.int8)
        for t_idx in range(TRAINING_SET_SIZE):
            data_in, data_out = self.training_set[t_idx]
            inputs[t_idx] = data_in
            outputs[t_idx] = data_out
        self._training_inputs = inputs
        self._training_outputs = outputs

    # ----------------------------------------------------------
    # Neighbor counting (ring topology)
    # ----------------------------------------------------------
    def get_actual_neighbor_count(self) -> int:
        pop = self.current_ann.population
        max_n = pop - 1
        return min(MAX_NEIGHBOR_NEURONS, max_n)

    def get_left_neighbor_count(self) -> int:
        actual = self.get_actual_neighbor_count()
        return (actual + 1) // 2

    def get_right_neighbor_count(self) -> int:
        return self.get_actual_neighbor_count() - self.get_left_neighbor_count()

    def get_synapse_start_index(self) -> int:
        return HALF_MAX_NEIGHBORS - self.get_left_neighbor_count()

    def get_synapse_end_index(self) -> int:
        return HALF_MAX_NEIGHBORS + self.get_right_neighbor_count()

    # ----------------------------------------------------------
    # Index conversion
    # ----------------------------------------------------------
    def buffer_index_to_offset(self, buf_idx: int) -> int:
        if buf_idx < HALF_MAX_NEIGHBORS:
            return buf_idx - HALF_MAX_NEIGHBORS  # negative (left)
        else:
            return buf_idx - HALF_MAX_NEIGHBORS + 1  # positive (right), skip 0

    def offset_to_buffer_index(self, offset: int) -> int:
        if offset == 0:
            return -1
        elif offset < 0:
            return HALF_MAX_NEIGHBORS + offset
        else:
            return HALF_MAX_NEIGHBORS + offset - 1

    def get_index_in_synapses_buffer(self, neighbor_offset: int) -> int:
        left_count = self.get_left_neighbor_count()
        right_count = self.get_right_neighbor_count()
        if (neighbor_offset == 0 or
                neighbor_offset < -left_count or
                neighbor_offset > right_count):
            return -1
        return self.offset_to_buffer_index(neighbor_offset)

    # ----------------------------------------------------------
    # Synapse access
    # ----------------------------------------------------------
    def get_synapse_weight(self, neuron_idx: int, buf_idx: int) -> int:
        return self.current_ann.synapses[neuron_idx * MAX_NEIGHBOR_NEURONS + buf_idx]

    def set_synapse_weight(self, neuron_idx: int, buf_idx: int, weight: int):
        self.current_ann.synapses[neuron_idx * MAX_NEIGHBOR_NEURONS + buf_idx] = weight

    # ----------------------------------------------------------
    # Ring topology
    # ----------------------------------------------------------
    def clamp_neuron_index(self, neuron_idx: int, value: int) -> int:
        pop = self.current_ann.population
        if value >= 0:
            nn_index = neuron_idx + value
        else:
            nn_index = neuron_idx + pop + value
        return nn_index % pop

    def get_neighbor_neuron_index(self, neuron_index: int, neighbor_offset: int) -> int:
        left_neighbors = self.get_left_neighbor_count()
        if neighbor_offset < left_neighbors:
            return self.clamp_neuron_index(
                neuron_index + neighbor_offset, -left_neighbors)
        else:
            return self.clamp_neuron_index(
                neuron_index + neighbor_offset + 1, -left_neighbors)

    # ----------------------------------------------------------
    # Tick simulation (the compute-heavy inner loop)
    # ----------------------------------------------------------
    def process_tick(self):
        """
        One tick of the ANN simulation.
        For each neuron, compute weighted sum from neighbors.
        """
        pop = self.current_ann.population
        neurons = self.current_ann.neurons

        # Clear buffer
        for i in range(pop):
            self.neuron_value_buffer[i] = 0

        start_idx = self.get_synapse_start_index()
        end_idx = self.get_synapse_end_index()

        for n in range(pop):
            neuron_value = neurons[n].value
            for m in range(start_idx, end_idx):
                weight = self.get_synapse_weight(n, m)
                offset = self.buffer_index_to_offset(m)
                nn_index = self.clamp_neuron_index(n, offset)
                self.neuron_value_buffer[nn_index] += weight * neuron_value

        # Clamp and update non-input neurons
        for n in range(pop):
            if neurons[n].type != NEURON_INPUT:
                neurons[n].value = clamp_neuron(self.neuron_value_buffer[n])

    # ----------------------------------------------------------
    # Neuron insertion and removal
    # ----------------------------------------------------------
    def insert_neuron(self, neuron_index: int, synapse_index: int):
        """Port of insertNeuron from score_addition.h."""
        syn = self.current_ann.synapses
        neurons = self.current_ann.neurons

        old_start = self.get_synapse_start_index()
        old_end = self.get_synapse_end_index()
        old_actual = self.get_actual_neighbor_count()
        old_left = self.get_left_neighbor_count()
        old_right = self.get_right_neighbor_count()

        assert old_start <= synapse_index < old_end

        # Save original neuron properties
        inserted_neuron = Neuron(NEURON_EVOLUTION, neurons[neuron_index].value)
        inserted_idx = neuron_index + 1

        synapse_full_idx = neuron_index * MAX_NEIGHBOR_NEURONS + synapse_index
        original_weight = syn[synapse_full_idx]

        pop = self.current_ann.population

        # Shift neurons and synapses right
        for i in range(pop, neuron_index, -1):
            neurons[i].type = neurons[i - 1].type
            neurons[i].value = neurons[i - 1].value
            neurons[i].mark_for_removal = neurons[i - 1].mark_for_removal
            src_start = (i - 1) * MAX_NEIGHBOR_NEURONS
            dst_start = i * MAX_NEIGHBOR_NEURONS
            syn[dst_start:dst_start + MAX_NEIGHBOR_NEURONS] = \
                syn[src_start:src_start + MAX_NEIGHBOR_NEURONS]

        neurons[inserted_idx].type = NEURON_EVOLUTION
        neurons[inserted_idx].value = inserted_neuron.value
        neurons[inserted_idx].mark_for_removal = False
        self.current_ann.population += 1

        # Recalculate after population change
        new_actual = self.get_actual_neighbor_count()
        new_start = self.get_synapse_start_index()
        new_end = self.get_synapse_end_index()

        # Zero out the inserted neuron's outgoing synapses
        ins_base = inserted_idx * MAX_NEIGHBOR_NEURONS
        for k in range(MAX_NEIGHBOR_NEURONS):
            syn[ins_base + k] = 0

        # Copy mutated outgoing synapse
        if synapse_index < HALF_MAX_NEIGHBORS:
            if synapse_index > new_start:
                syn[ins_base + synapse_index - 1] = original_weight
        else:
            syn[ins_base + synapse_index] = original_weight

        # Update neighbor synapses
        for delta in range(-old_left, old_right + 1):
            if delta == 0:
                continue
            updated_idx = self.clamp_neuron_index(inserted_idx, delta)

            # Find inserted neuron in neighbor list
            inserted_in_neighbor = -1
            for k in range(new_actual):
                nn_idx = self.get_neighbor_neuron_index(updated_idx, k)
                if nn_idx == inserted_idx:
                    inserted_in_neighbor = new_start + k

            assert inserted_in_neighbor >= 0

            upd_base = updated_idx * MAX_NEIGHBOR_NEURONS

            if delta < 0:
                # Shift right side
                for k in range(new_end - 1, inserted_in_neighbor, -1):
                    syn[upd_base + k] = syn[upd_base + k - 1]
                if delta == -1:
                    syn[upd_base + inserted_in_neighbor] = 0
            else:
                # Shift left side
                for k in range(new_start, inserted_in_neighbor):
                    syn[upd_base + k] = syn[upd_base + k + 1]

    def remove_neuron(self, neuron_idx: int):
        """Port of removeNeuron from score_addition.h."""
        syn = self.current_ann.synapses
        neurons = self.current_ann.neurons

        left_count = self.get_left_neighbor_count()
        right_count = self.get_right_neighbor_count()
        start_idx = self.get_synapse_start_index()
        end_idx = self.get_synapse_end_index()

        # Remove incoming synapses from neighbors
        for offset in range(-left_count, right_count + 1):
            if offset == 0:
                continue
            nn_idx = self.clamp_neuron_index(neuron_idx, offset)
            syn_idx = self.get_index_in_synapses_buffer(-offset)
            if syn_idx < 0:
                continue

            nn_base = nn_idx * MAX_NEIGHBOR_NEURONS
            if syn_idx >= HALF_MAX_NEIGHBORS:
                # Right side: shift left
                for k in range(syn_idx, end_idx - 1):
                    syn[nn_base + k] = syn[nn_base + k + 1]
                syn[nn_base + end_idx - 1] = 0
            else:
                # Left side: shift right
                for k in range(syn_idx, start_idx, -1):
                    syn[nn_base + k] = syn[nn_base + k - 1]
                syn[nn_base + start_idx] = 0

        # Shift arrays
        pop = self.current_ann.population
        for shift_idx in range(neuron_idx, pop - 1):
            neurons[shift_idx].type = neurons[shift_idx + 1].type
            neurons[shift_idx].value = neurons[shift_idx + 1].value
            neurons[shift_idx].mark_for_removal = neurons[shift_idx + 1].mark_for_removal
            src = (shift_idx + 1) * MAX_NEIGHBOR_NEURONS
            dst = shift_idx * MAX_NEIGHBOR_NEURONS
            syn[dst:dst + MAX_NEIGHBOR_NEURONS] = syn[src:src + MAX_NEIGHBOR_NEURONS]
        self.current_ann.population -= 1

    def scan_redundant_neurons(self) -> int:
        """Check for neurons with all-zero incoming or outgoing synapses."""
        pop = self.current_ann.population
        syn = self.current_ann.synapses
        neurons = self.current_ann.neurons

        start_idx = self.get_synapse_start_index()
        end_idx = self.get_synapse_end_index()
        left_count = self.get_left_neighbor_count()
        right_count = self.get_right_neighbor_count()

        count = 0
        for i in range(pop):
            neurons[i].mark_for_removal = False
            if neurons[i].type == NEURON_EVOLUTION:
                # Check outgoing
                all_out_zero = True
                base = i * MAX_NEIGHBOR_NEURONS
                for m in range(start_idx, end_idx):
                    if syn[base + m] != 0:
                        all_out_zero = False
                        break

                # Check incoming
                all_in_zero = True
                for offset in range(-left_count, right_count + 1):
                    if offset == 0:
                        continue
                    nn_idx = self.clamp_neuron_index(i, offset)
                    syn_idx = self.get_index_in_synapses_buffer(-offset)
                    if syn_idx < 0:
                        continue
                    nn_base = nn_idx * MAX_NEIGHBOR_NEURONS
                    if syn[nn_base + syn_idx] != 0:
                        all_in_zero = False
                        break

                if all_out_zero or all_in_zero:
                    neurons[i].mark_for_removal = True
                    count += 1
        return count

    def clean_ann(self):
        """Remove neurons marked for removal."""
        neurons = self.current_ann.neurons
        idx = 0
        while idx < self.current_ann.population:
            if neurons[idx].mark_for_removal:
                self.remove_neuron(idx)
            else:
                idx += 1

    # ----------------------------------------------------------
    # Mutation
    # ----------------------------------------------------------
    def mutate(self, step: int):
        """Port of mutate() from score_addition.h."""
        pop = self.current_ann.population
        actual_neighbors = self.get_actual_neighbor_count()
        syn = self.current_ann.synapses

        mutation_val = self.init_value.synapse_mutation[step]
        total_valid = pop * actual_neighbors
        flat_idx = (mutation_val >> 1) % total_valid

        neuron_idx = flat_idx // actual_neighbors
        local_syn_idx = flat_idx % actual_neighbors

        synapse_index = local_syn_idx + self.get_synapse_start_index()
        full_buf_idx = neuron_idx * MAX_NEIGHBOR_NEURONS + synapse_index

        weight_change = -1 if (mutation_val & 1) == 0 else 1
        new_weight = syn[full_buf_idx] + weight_change

        if -1 <= new_weight <= 1:
            syn[full_buf_idx] = new_weight
        else:
            self.insert_neuron(neuron_idx, synapse_index)

        while self.scan_redundant_neurons() > 0:
            self.clean_ann()

    # ----------------------------------------------------------
    # Training data loading & inference
    # ----------------------------------------------------------
    def load_training_data(self, training_index: int):
        """Load one training pair into the ANN."""
        pop = self.current_ann.population
        neurons = self.current_ann.neurons
        data_in, data_out = self.training_set[training_index]

        input_idx = 0
        for n in range(pop):
            neurons[n].value = 0
            if neurons[n].type == NEURON_INPUT:
                neurons[n].value = data_in[input_idx]
                input_idx += 1

        self.output_neuron_expected_value[:] = data_out

    def run_tick_simulation(self, training_index: int):
        """Run tick simulation for one training pair."""
        pop = self.current_ann.population
        neurons = self.current_ann.neurons

        self.load_training_data(training_index)

        # Save initial values
        for i in range(pop):
            self.previous_neuron_value[i] = neurons[i].value

        for tick in range(NUMBER_OF_TICKS):
            self.process_tick()

            all_unchanged = True
            all_output_nonzero = True

            for n in range(pop):
                if self.previous_neuron_value[n] != neurons[n].value:
                    all_unchanged = False
                if neurons[n].type == NEURON_OUTPUT and neurons[n].value == 0:
                    all_output_nonzero = False

            if all_output_nonzero or all_unchanged:
                break

            for n in range(pop):
                self.previous_neuron_value[n] = neurons[n].value

    def compute_matching_output(self) -> int:
        """Count matching output bits."""
        pop = self.current_ann.population
        neurons = self.current_ann.neurons
        r = 0
        out_idx = 0
        for i in range(pop):
            if neurons[i].type == NEURON_OUTPUT:
                if neurons[i].value == self.output_neuron_expected_value[out_idx]:
                    r += 1
                out_idx += 1
        return r

    def infer_ann(self) -> int:
        """Run inference on all training pairs and return total score."""
        # Use JAX batch path when available (TPU/GPU)
        if self.use_jax:
            return self._batch_infer_ann_jax()
        # Fallback: sequential Python path
        score = 0
        for i in range(TRAINING_SET_SIZE):
            self.run_tick_simulation(i)
            score += self.compute_matching_output()
        return score

    def _batch_infer_ann_jax(self) -> int:
        """JAX-accelerated inference over all 16,384 training pairs."""
        pop = self.current_ann.population
        neurons = self.current_ann.neurons

        # Build weight matrix from current ANN topology
        W_np = np.zeros((pop, pop), dtype=np.float32)
        start_idx = self.get_synapse_start_index()
        end_idx = self.get_synapse_end_index()
        for n in range(pop):
            for m in range(start_idx, end_idx):
                weight = self.get_synapse_weight(n, m)
                if weight != 0:
                    offset = self.buffer_index_to_offset(m)
                    nn_idx = self.clamp_neuron_index(n, offset)
                    W_np[nn_idx, n] += weight
        W = jnp.array(W_np)

        # Build neuron type and index arrays
        neuron_types_list = [neurons[i].type for i in range(pop)]
        is_input = np.array([1 if t == NEURON_INPUT else 0 for t in neuron_types_list], dtype=np.float32)
        is_input_j = jnp.array(is_input)

        output_indices = [i for i in range(pop) if neurons[i].type == NEURON_OUTPUT]
        output_indices_arr = jnp.array(output_indices)

        # Build batch input values: [16384, pop]
        batch_values_np = np.zeros((TRAINING_SET_SIZE, pop), dtype=np.float32)
        input_neuron_indices = [i for i in range(pop) if neurons[i].type == NEURON_INPUT]
        for col_idx, neuron_idx in enumerate(input_neuron_indices):
            batch_values_np[:, neuron_idx] = self._training_inputs[:, col_idx].astype(np.float32)

        expected_outputs = jnp.array(self._training_outputs)
        values = jnp.array(batch_values_np)

        # Run ticks (batched matmul on TPU)
        for tick in range(NUMBER_OF_TICKS):
            # new_values = clamp(W @ values^T)^T, keep input neurons
            new_vals = jnp.matmul(values, W.T)  # [batch, pop]
            new_vals = jnp.clip(new_vals, -1.0, 1.0)
            # Preserve input neurons
            new_vals = jnp.where(is_input_j, values, new_vals)

            # Early exit: all outputs non-zero or no change
            outputs = new_vals[:, output_indices_arr]
            if bool(jnp.all(outputs != 0)) or bool(jnp.array_equal(new_vals, values)):
                values = new_vals
                break
            values = new_vals

        # Count matching outputs
        final_outputs = values[:, output_indices_arr]
        matches = jnp.sum(final_outputs == expected_outputs)
        return int(matches)

    # ----------------------------------------------------------
    # ANN initialization
    # ----------------------------------------------------------
    def initialize_ann(self, public_key: bytes, nonce: bytes) -> int:
        """
        Initialize the ANN from public key + nonce.
        Port of initializeANN() from score_addition.h.
        """
        # Hash public_key || nonce
        combined = public_key[:32] + nonce[:32]
        hash_out = kangaroo_twelve(combined, 32)

        # Compute InitValue size
        # outputNeuronPositions: 8 uint64
        # synapseWeight: PADDING_NUMBER_OF_SYNAPSES // 32 uint64s
        # synapseMutation: 100 uint64s
        iv_size = (NUMBER_OF_OUTPUT_NEURONS * 8 +
                   (PADDING_NUMBER_OF_SYNAPSES // 32) * 8 +
                   NUMBER_OF_MUTATIONS * 8)

        iv_bytes = random2(hash_out, self.pool, iv_size)
        self.init_value = parse_init_value(iv_bytes)

        # Initialize population
        pop = NUMBER_OF_NEURONS  # 22
        self.current_ann.population = pop

        # Generate training set
        self.training_set = generate_training_set()

        # All start as input neurons
        neuron_indices = list(range(pop))
        for i in range(pop):
            self.current_ann.neurons[i].type = NEURON_INPUT

        # Randomly place output neurons
        neuron_count = pop
        for i in range(NUMBER_OF_OUTPUT_NEURONS):
            idx = self.init_value.output_neuron_positions[i] % neuron_count
            self.current_ann.neurons[neuron_indices[idx]].type = NEURON_OUTPUT

            neuron_count -= 1
            neuron_indices[idx] = neuron_indices[neuron_count]

        # Initialize synapse weights from packed random values
        for i in range(MAX_NUMBER_OF_SYNAPSES // 32):
            packed = self.init_value.synapse_weight[i]
            for j in range(32):
                extract_val = (packed >> (j * 2)) & 0b11
                if extract_val == 2:
                    w = -1
                elif extract_val == 3:
                    w = 1
                else:
                    w = 0
                self.current_ann.synapses[32 * i + j] = w

        # Handle remainder
        remainder = MAX_NUMBER_OF_SYNAPSES % 32
        if remainder > 0:
            last_block = MAX_NUMBER_OF_SYNAPSES // 32
            packed = self.init_value.synapse_weight[last_block]
            for j in range(remainder):
                extract_val = (packed >> (j * 2)) & 0b11
                if extract_val == 2:
                    w = -1
                elif extract_val == 3:
                    w = 1
                else:
                    w = 0
                self.current_ann.synapses[32 * last_block + j] = w

        # Run first inference
        score = self.infer_ann()
        return score

    # ----------------------------------------------------------
    # Main mining function
    # ----------------------------------------------------------
    def compute_score(self, public_key: bytes, nonce: bytes) -> int:
        """
        Compute PoW score for a given nonce.
        Port of computeScore() from score_addition.h.
        """
        best_r = self.initialize_ann(public_key, nonce)
        self.best_ann.copy_from(self.current_ann)

        for s in range(NUMBER_OF_MUTATIONS):
            self.mutate(s)

            if self.current_ann.population >= POPULATION_THRESHOLD:
                break

            r = self.infer_ann()

            if r >= best_r:
                best_r = r
                self.best_ann.copy_from(self.current_ann)
            else:
                self.current_ann.copy_from(self.best_ann)

        return best_r

    def find_solution(self, public_key: bytes, nonce: bytes) -> bool:
        """Check if a nonce produces a solution."""
        score = self.compute_score(public_key, nonce)
        return score >= SOLUTION_THRESHOLD


# ============================================================
# Batch inference with JAX (TPU-optimized)
# ============================================================
if HAS_JAX:
    def build_weight_matrix(miner: AdditionMiner) -> np.ndarray:
        """
        Build the weight matrix W for the current ANN.
        W[target, source] = synapse weight from source -> target.
        This converts the ring topology into a dense matrix.
        """
        pop = miner.current_ann.population
        W = np.zeros((pop, pop), dtype=np.int8)

        start_idx = miner.get_synapse_start_index()
        end_idx = miner.get_synapse_end_index()

        for n in range(pop):
            for m in range(start_idx, end_idx):
                weight = miner.get_synapse_weight(n, m)
                if weight != 0:
                    offset = miner.buffer_index_to_offset(m)
                    nn_idx = miner.clamp_neuron_index(n, offset)
                    # source=n pushes to target=nn_idx
                    W[nn_idx, n] += weight

        return W

    @jax.jit
    def batch_tick(W, neuron_values, neuron_types):
        """
        One tick for a batch of training pairs.
        W: [pop, pop] weight matrix
        neuron_values: [batch, pop] current values
        neuron_types: [pop] neuron types
        Returns: [batch, pop] updated values
        """
        # Matrix multiply: new_values = W @ values^T -> [pop, batch]
        new_values = jnp.matmul(W.astype(jnp.int32),
                                neuron_values.astype(jnp.int32).T).T
        # Clamp to [-1, 1]
        new_values = jnp.clip(new_values, -1, 1)
        # Keep input neurons unchanged
        is_input = (neuron_types == NEURON_INPUT)
        new_values = jnp.where(is_input, neuron_values, new_values)
        return new_values.astype(jnp.int8)

    def batch_infer_ann(miner: AdditionMiner) -> int:
        """
        JAX-accelerated inference over all training pairs.
        Replaces the sequential infer_ann() with batched matmul.
        """
        pop = miner.current_ann.population
        neurons = miner.current_ann.neurons

        # Build weight matrix
        W = jnp.array(build_weight_matrix(miner), dtype=jnp.int8)

        # Neuron types
        neuron_types = jnp.array([neurons[i].type for i in range(pop)], dtype=jnp.int8)

        # Prepare all training data as batch
        batch_values = np.zeros((TRAINING_SET_SIZE, pop), dtype=np.int8)
        expected_outputs = np.zeros((TRAINING_SET_SIZE, NUMBER_OF_OUTPUT_NEURONS), dtype=np.int8)

        # Find output neuron indices
        output_indices = []
        for i in range(pop):
            if neurons[i].type == NEURON_OUTPUT:
                output_indices.append(i)

        for t_idx in range(TRAINING_SET_SIZE):
            data_in, data_out = miner.training_set[t_idx]
            input_idx = 0
            for n in range(pop):
                if neurons[n].type == NEURON_INPUT:
                    batch_values[t_idx, n] = data_in[input_idx]
                    input_idx += 1
            expected_outputs[t_idx] = data_out

        batch_values = jnp.array(batch_values)
        expected_outputs = jnp.array(expected_outputs)
        output_indices_arr = jnp.array(output_indices)

        # Run ticks
        prev_values = batch_values
        for tick in range(NUMBER_OF_TICKS):
            new_values = batch_tick(W, prev_values, neuron_types)

            # Check convergence (all outputs non-zero OR no change)
            outputs = new_values[:, output_indices_arr]
            all_nonzero = jnp.all(outputs != 0, axis=1)  # per sample
            unchanged = jnp.array_equal(new_values, prev_values)

            if bool(jnp.all(all_nonzero)) or bool(unchanged):
                prev_values = new_values
                break
            prev_values = new_values

        # Count matching outputs
        final_outputs = prev_values[:, output_indices_arr]
        matches = jnp.sum(final_outputs == expected_outputs)
        return int(matches)
