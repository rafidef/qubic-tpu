"""
Tests for the Qubic TPU Miner scoring algorithm.
Ports key test cases from score_addition_test.cpp to verify correctness.

Run with: python test_score.py
"""
import sys
import struct
import time

from qubic_k12 import kangaroo_twelve, keccak_p1600_permute_12, random2
from qubic_score import (
    clamp_neuron, to_ternary_bits, AdditionMiner,
    NEURON_INPUT, NEURON_OUTPUT, NEURON_EVOLUTION,
    MAX_NEIGHBOR_NEURONS, NUMBER_OF_INPUT_NEURONS,
    NUMBER_OF_OUTPUT_NEURONS, NUMBER_OF_MUTATIONS,
    TRAINING_SET_SIZE, generate_training_set,
)

PASS = 0
FAIL = 0


def check(condition, name):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name}")


# ============================================================
# Test fixtures (matching score_addition_test.cpp)
# ============================================================
def make_test_seeds():
    """Deterministic seeds matching the C++ test fixture."""
    mining_seed = bytes(range(32))
    public_key = bytes(range(32, 64))
    nonce = bytes(range(64, 96))
    return mining_seed, public_key, nonce


def make_test_miner(pool=None):
    """Create a miner with test seeds."""
    mining_seed, public_key, nonce = make_test_seeds()
    miner = AdditionMiner()
    if pool is None:
        # For helper/unit tests, we don't need the full pool.
        # We'll create a small dummy pool that works for initialization.
        # The real pool is 512MB; for tests, we use a smaller one.
        pool = bytearray(1024 * 1024)  # 1MB dummy
        state = bytearray(200)
        state[:32] = mining_seed[:32]
        for i in range(0, len(pool), 200):
            keccak_p1600_permute_12(state)
            end = min(i + 200, len(pool))
            pool[i:end] = state[:end - i]
    miner.initialize(mining_seed, pool)
    return miner, pool


# ============================================================
# Helper function tests
# ============================================================
def test_clamp_neuron():
    print("Testing clampNeuron...")
    check(clamp_neuron(0) == 0, "clamp(0) == 0")
    check(clamp_neuron(1) == 1, "clamp(1) == 1")
    check(clamp_neuron(-1) == -1, "clamp(-1) == -1")
    check(clamp_neuron(2) == 1, "clamp(2) == 1")
    check(clamp_neuron(100) == 1, "clamp(100) == 1")
    check(clamp_neuron(-2) == -1, "clamp(-2) == -1")
    check(clamp_neuron(-100) == -1, "clamp(-100) == -1")


def test_ternary_bits():
    print("Testing toTenaryBits...")
    # 0 -> all -1
    bits = to_ternary_bits(0, 7)
    check(all(b == -1 for b in bits), "0 -> all -1")

    # 5 = 101 -> [1, -1, 1, -1, -1, -1, -1]
    bits = to_ternary_bits(5, 7)
    check(bits[0] == 1, "5 bit0 == 1")
    check(bits[1] == -1, "5 bit1 == -1")
    check(bits[2] == 1, "5 bit2 == 1")
    for i in range(3, 7):
        check(bits[i] == -1, f"5 bit{i} == -1")

    # -1 -> all 1s
    bits = to_ternary_bits(-1, 7)
    check(all(b == 1 for b in bits), "-1 -> all 1")


def test_training_set():
    print("Testing training set generation...")
    ts = generate_training_set()
    check(len(ts) == TRAINING_SET_SIZE, f"training set size == {TRAINING_SET_SIZE}")

    # Check a known pair: A=0, B=0, C=0
    # A=0 -> bits = [-1,-1,-1,-1,-1,-1,-1], B=0 -> same
    # C=0 -> bits = [-1,-1,-1,-1,-1,-1,-1,-1]
    # Index: A goes from -64 to 63, B from -64 to 63
    # A=0 is at position 64 in the A loop, B=0 at position 64
    # Index = 64 * 128 + 64 = 8256
    inp, out = ts[8256]
    a_bits = to_ternary_bits(0, 7)
    b_bits = to_ternary_bits(0, 7)
    c_bits = to_ternary_bits(0, 8)
    check(inp == a_bits + b_bits, "pair(0,0) input correct")
    check(out == c_bits, "pair(0,0) output correct")


# ============================================================
# Neighbor counting tests
# ============================================================
def test_neighbor_counting():
    print("Testing neighbor counting...")
    miner, _ = make_test_miner()

    # Pop < MAX_NEIGHBORS -> fully connected
    miner.current_ann.population = MAX_NEIGHBOR_NEURONS // 2
    check(miner.get_actual_neighbor_count() == MAX_NEIGHBOR_NEURONS // 2 - 1,
          "small pop: actual == pop-1")

    # Pop > MAX_NEIGHBORS -> capped
    miner.current_ann.population = MAX_NEIGHBOR_NEURONS * 2
    check(miner.get_actual_neighbor_count() == MAX_NEIGHBOR_NEURONS,
          "large pop: actual == MAX")

    # Left/right split
    miner.current_ann.population = 23  # 22 neighbors
    check(miner.get_actual_neighbor_count() == 22, "pop=23: actual=22")
    check(miner.get_left_neighbor_count() == 11, "pop=23: left=11")
    check(miner.get_right_neighbor_count() == 11, "pop=23: right=11")

    miner.current_ann.population = 22  # 21 neighbors
    check(miner.get_actual_neighbor_count() == 21, "pop=22: actual=21")
    check(miner.get_left_neighbor_count() == 11, "pop=22: left=11")
    check(miner.get_right_neighbor_count() == 10, "pop=22: right=10")

    # Left + right always == actual
    for pop in range(22, 200):
        miner.current_ann.population = pop
        left = miner.get_left_neighbor_count()
        right = miner.get_right_neighbor_count()
        actual = miner.get_actual_neighbor_count()
        check(left + right == actual, f"pop={pop}: left+right==actual")


# ============================================================
# Index conversion tests
# ============================================================
def test_buffer_index_offset():
    print("Testing buffer index <-> offset conversion...")
    miner, _ = make_test_miner()
    miner.current_ann.population = 22
    half_max = MAX_NEIGHBOR_NEURONS // 2  # 364

    # Left side
    check(miner.buffer_index_to_offset(363) == -1, "buf 363 -> offset -1")
    check(miner.buffer_index_to_offset(354) == -10, "buf 354 -> offset -10")
    check(miner.buffer_index_to_offset(0) == -364, "buf 0 -> offset -364")

    check(miner.offset_to_buffer_index(-1) == 363, "offset -1 -> buf 363")
    check(miner.offset_to_buffer_index(-10) == 354, "offset -10 -> buf 354")
    check(miner.offset_to_buffer_index(-364) == 0, "offset -364 -> buf 0")

    # Right side
    check(miner.buffer_index_to_offset(364) == 1, "buf 364 -> offset 1")
    check(miner.buffer_index_to_offset(373) == 10, "buf 373 -> offset 10")
    check(miner.buffer_index_to_offset(727) == 364, "buf 727 -> offset 364")

    check(miner.offset_to_buffer_index(1) == 364, "offset 1 -> buf 364")
    check(miner.offset_to_buffer_index(10) == 373, "offset 10 -> buf 373")
    check(miner.offset_to_buffer_index(364) == 727, "offset 364 -> buf 727")

    # Center
    check(miner.buffer_index_to_offset(half_max) == 1, "buf half -> offset 1")
    check(miner.buffer_index_to_offset(half_max - 1) == -1, "buf half-1 -> offset -1")
    check(miner.offset_to_buffer_index(0) == -1, "offset 0 -> invalid")

    # Round-trip
    for idx in range(MAX_NEIGHBOR_NEURONS):
        offset = miner.buffer_index_to_offset(idx)
        check(offset != 0, f"buf {idx} offset != 0")
        back = miner.offset_to_buffer_index(offset)
        check(back == idx, f"round-trip buf {idx}")


# ============================================================
# Clamp neuron index tests
# ============================================================
def test_clamp_neuron_index():
    print("Testing clampNeuronIndex (ring wrapping)...")
    miner, _ = make_test_miner()
    miner.current_ann.population = 22

    check(miner.clamp_neuron_index(0, 5) == 5, "0+5=5")
    check(miner.clamp_neuron_index(10, 5) == 15, "10+5=15")
    check(miner.clamp_neuron_index(10, -5) == 5, "10-5=5")
    check(miner.clamp_neuron_index(5, -3) == 2, "5-3=2")

    # Wrap around
    check(miner.clamp_neuron_index(20, 5) == 3, "20+5=3 (wrap)")
    check(miner.clamp_neuron_index(21, 1) == 0, "21+1=0 (wrap)")
    check(miner.clamp_neuron_index(0, -1) == 21, "0-1=21 (wrap)")
    check(miner.clamp_neuron_index(2, -5) == 19, "2-5=19 (wrap)")


# ============================================================
# KangarooTwelve hash test
# ============================================================
def test_k12_basic():
    print("Testing KangarooTwelve basic...")
    # Test with empty input
    result = kangaroo_twelve(b'', 32)
    check(len(result) == 32, "K12 empty -> 32 bytes")

    # Test determinism
    data = b"Hello, Qubic!"
    h1 = kangaroo_twelve(data, 32)
    h2 = kangaroo_twelve(data, 32)
    check(h1 == h2, "K12 deterministic")

    # Different input -> different output
    h3 = kangaroo_twelve(b"Different input", 32)
    check(h1 != h3, "K12 different inputs -> different outputs")


# ============================================================
# Process tick test
# ============================================================
def test_process_tick():
    print("Testing processTick (input neurons preserved)...")
    miner, pool = make_test_miner()
    _, public_key, nonce = make_test_seeds()

    # Initialize ANN with a small dummy pool
    # We need to test with a properly initialized ANN
    # Use direct initialization instead
    miner.current_ann.population = 22
    for i in range(22):
        if i < 14:
            miner.current_ann.neurons[i].type = NEURON_INPUT
            miner.current_ann.neurons[i].value = 1
        else:
            miner.current_ann.neurons[i].type = NEURON_OUTPUT
            miner.current_ann.neurons[i].value = 0

    miner.process_tick()

    # Verify input neurons unchanged
    for i in range(14):
        check(miner.current_ann.neurons[i].value == 1,
              f"input neuron {i} preserved after tick")


# ============================================================
# Insert neuron test
# ============================================================
def test_insert_neuron():
    print("Testing insertNeuron...")
    miner, pool = make_test_miner()

    # Set up simple ANN
    miner.current_ann.population = 22
    for i in range(22):
        miner.current_ann.neurons[i].type = NEURON_INPUT if i < 14 else NEURON_OUTPUT
        miner.current_ann.neurons[i].value = 0
        miner.current_ann.neurons[i].mark_for_removal = False

    # Zero all synapses
    for i in range(len(miner.current_ann.synapses)):
        miner.current_ann.synapses[i] = 0

    old_pop = miner.current_ann.population
    start_idx = miner.get_synapse_start_index()

    # Count types before
    inputs_before = sum(1 for i in range(old_pop)
                        if miner.current_ann.neurons[i].type == NEURON_INPUT)
    outputs_before = sum(1 for i in range(old_pop)
                         if miner.current_ann.neurons[i].type == NEURON_OUTPUT)

    miner.insert_neuron(0, start_idx)

    check(miner.current_ann.population == old_pop + 1, "population increased by 1")
    check(miner.current_ann.neurons[1].type == NEURON_EVOLUTION,
          "inserted neuron is EVOLUTION type")

    # Count types after
    pop = miner.current_ann.population
    inputs_after = sum(1 for i in range(pop)
                       if miner.current_ann.neurons[i].type == NEURON_INPUT)
    outputs_after = sum(1 for i in range(pop)
                        if miner.current_ann.neurons[i].type == NEURON_OUTPUT)

    check(inputs_after == inputs_before, "input count unchanged")
    check(outputs_after == outputs_before, "output count unchanged")


# ============================================================
# Pipeline test (matching C++ "smallset" test)
# ============================================================
def test_pipeline_smallset():
    print("Testing pipeline (smallset)...")
    miner, pool = make_test_miner()

    # Set up manually (matching C++ test)
    miner.current_ann.population = 22
    for i in range(22):
        miner.current_ann.neurons[i].value = 0
        miner.current_ann.neurons[i].mark_for_removal = False
        miner.current_ann.neurons[i].type = NEURON_INPUT if i < 14 else NEURON_OUTPUT
    for i in range(len(miner.current_ann.synapses)):
        miner.current_ann.synapses[i] = 0
    for i in range(14):
        miner.current_ann.neurons[i].value = 1

    # Case1: no insertion
    # synapse: neuron 0 -> neuron 14 (offset -8)
    # offsetToBufferIndex(-8) = 364 + (-8) = 356
    # localSynapseIdx = 356 - startIdx(353) = 3
    # flatIdx = 0 * 21 + 3 = 3
    # mutationValue = (3 << 1) | 1 = 7
    miner.init_value.synapse_mutation[0] = 7

    buf_idx = miner.offset_to_buffer_index(-8)
    check(miner.get_synapse_weight(0, buf_idx) == 0,
          "synapse 0->14 initially 0")

    miner.mutate(0)

    check(miner.get_synapse_weight(0, buf_idx) == 1,
          "synapse 0->14 now 1 after mutation")
    check(miner.current_ann.population == 22,
          "population unchanged (no insertion)")

    # Process tick
    miner.process_tick()

    check(miner.current_ann.neurons[14].value == 1,
          "neuron 14 receives 1*1=1")
    for i in range(14):
        check(miner.current_ann.neurons[i].value == 1,
              f"input neuron {i} unchanged after tick")


# ============================================================
# Score computation test (correctness check)
# ============================================================
def test_compute_score():
    print("Testing computeScore (deterministic)...")
    # This test verifies that compute_score produces consistent results
    # Note: without the full 512MB pool, results won't match C++ exactly
    # But they should be deterministic within our Python implementation

    miner, pool = make_test_miner()
    _, public_key, nonce = make_test_seeds()

    # Use a test nonce with odd first byte (for addition algorithm)
    test_nonce = bytearray(nonce)
    test_nonce[0] |= 1

    t0 = time.time()
    score = miner.compute_score(public_key, bytes(test_nonce))
    t1 = time.time()

    print(f"  Score: {score}")
    print(f"  Time: {t1 - t0:.2f}s")

    # Score should be a reasonable value
    check(0 <= score <= TRAINING_SET_SIZE * NUMBER_OF_OUTPUT_NEURONS,
          f"score in valid range [0, {TRAINING_SET_SIZE * NUMBER_OF_OUTPUT_NEURONS}]")

    # Determinism: same inputs -> same score
    miner2, _ = make_test_miner(pool)
    score2 = miner2.compute_score(public_key, bytes(test_nonce))
    check(score == score2, "compute_score is deterministic")


# ============================================================
# Run all tests
# ============================================================
def run_tests():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("\n" + "=" * 60)
    print("  Qubic TPU Miner — Test Suite")
    print("=" * 60 + "\n")

    test_clamp_neuron()
    test_ternary_bits()
    test_training_set()
    test_neighbor_counting()
    test_buffer_index_offset()
    test_clamp_neuron_index()
    test_k12_basic()
    test_process_tick()
    test_insert_neuron()
    test_pipeline_smallset()

    print("\n--- Slow tests (scoring) ---")
    test_compute_score()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    print("\nAll tests passed! ✓")


if __name__ == '__main__':
    run_tests()
