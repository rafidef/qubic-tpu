"""
Qubic TPU Miner — Custom Runner for qli-Client
Implements the Qubic/Qiner protocol as documented in CustomRunner.md.

The qli-Client starts this runner with:
    python tpu_miner.py <IP> <ID> <THREADS>

And parses STDOUT for: "Y it/s" and "X solutions"

Solution delivery uses the standard Qiner TCP protocol on port 21841.
"""
import sys
import os
import time
import struct
import socket
import threading
import secrets
import signal
from datetime import datetime, timezone

from qubic_k12 import kangaroo_twelve
from qubic_keys import (
    get_public_key_from_identity,
    get_subseed_from_seed,
    get_private_key_from_subseed,
)
from qubic_score import (
    AdditionMiner, SOLUTION_THRESHOLD,
    TRAINING_SET_SIZE, NUMBER_OF_OUTPUT_NEURONS, HAS_JAX,
)


# ============================================================
# Constants
# ============================================================
BROADCAST_MESSAGE = 1
NODE_PORT = 21841  # Default Qubic node port
PACKET_HEADER_SIZE = 8


# ============================================================
# Global state (matching Qiner.cpp structure)
# ============================================================
class GlobalState:
    def __init__(self):
        self.state = 0  # 0=running, 1=shutdown requested
        self.computor_public_key = bytes(32)
        self.random_seed = bytes(32)
        self.number_of_mining_iterations = 0
        self.number_of_found_solutions = 0
        self.found_nonces = []
        self.lock = threading.Lock()
        self.mining_id = ""

g = GlobalState()


# ============================================================
# Signal handler (matching Qiner.cpp ctrlCHandlerRoutine)
# ============================================================
def signal_handler(sig, frame):
    if g.state == 0:
        g.state = 1
    else:
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)


# ============================================================
# Network: Solution packet builder
# ============================================================
def build_solution_packet(
    nonce: bytes,
    mining_seed: bytes,
    computor_public_key: bytes,
) -> bytes:
    """
    Build the solution packet matching the Qiner protocol.
    
    Packet structure:
        RequestResponseHeader (8 bytes)
        Message:
            sourcePublicKey  (32 bytes) — signing public key
            destPublicKey    (32 bytes) — computor public key
            gammingNonce     (32 bytes) — random, chosen so gammingKey[0] == 0
        solutionMiningSeed   (32 bytes) — XOR encrypted
        solutionNonce        (32 bytes) — XOR encrypted
        signature            (64 bytes) — FourQ signature (placeholder for now)
    Total: 8 + 32 + 32 + 32 + 32 + 32 + 64 = 232 bytes

    NOTE: Without the FourQ elliptic curve library, we cannot produce a valid
    signature. The qli-Client handles solution delivery itself when using
    the custom runner protocol — the runner just reports solutions via STDOUT.
    The qli-Client picks them up and submits them through its own secure channel.
    """
    # For the custom runner protocol, the qli-Client handles solution delivery.
    # We just need to report solutions via STDOUT.
    # However, if direct submission is needed, here's the packet structure:

    packet = bytearray(232)

    # Header
    packet[0] = 232 & 0xFF         # size[0]
    packet[1] = (232 >> 8) & 0xFF  # size[1]
    packet[2] = (232 >> 16) & 0xFF # size[2]
    packet[3] = BROADCAST_MESSAGE   # type

    # Dejavu (random non-zero)
    dejavu = struct.unpack('<I', secrets.token_bytes(4))[0]
    if dejavu == 0:
        dejavu = 1
    struct.pack_into('<I', packet, 4, dejavu)

    # Message: source = computor key (simplified — need signing key in practice)
    packet[8:40] = computor_public_key[:32]    # sourcePublicKey
    packet[40:72] = computor_public_key[:32]   # destPublicKey

    # GammingNonce: find one where K12(sharedKey || gammingNonce)[0] == 0
    shared_key = bytes(32)  # zeros for non-computor case
    for attempt in range(10000):
        gamming_nonce = secrets.token_bytes(32)
        combined = shared_key + gamming_nonce
        gamming_key = kangaroo_twelve(combined, 32)
        if gamming_key[0] == 0:
            packet[72:104] = gamming_nonce
            # Encrypt solution
            gamma = kangaroo_twelve(gamming_key, 64)
            for i in range(32):
                packet[104 + i] = mining_seed[i] ^ gamma[i]
                packet[136 + i] = nonce[i] ^ gamma[32 + i]
            break

    # Signature placeholder (64 bytes of zeros — needs FourQ)
    # In practice, the qli-Client handles signing for custom runners

    return bytes(packet)


def submit_solution(nonce: bytes, node_ip: str):
    """Submit a found solution to the Qubic node."""
    try:
        packet = build_solution_packet(
            nonce, g.random_seed, g.computor_public_key)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((node_ip, NODE_PORT))
        sock.sendall(packet)
        sock.close()
        return True
    except Exception as e:
        return False


# ============================================================
# Mining thread (matching Qiner.cpp miningThreadProc)
# ============================================================
def mining_thread_proc(pool: bytes):
    """Single mining thread — matches Qiner.cpp miningThreadProc."""
    miner = AdditionMiner()
    miner.initialize(g.random_seed, pool)

    while g.state == 0:
        # Generate random nonce
        nonce = bytearray(secrets.token_bytes(32))

        solution_found = False

        # First byte determines score type (matching C++ logic)
        if (nonce[0] & 1) == 0:
            # Hyperidentity — not implemented yet, skip
            pass
        else:
            # Addition
            solution_found = miner.find_solution(
                g.computor_public_key, bytes(nonce))

        if solution_found:
            with g.lock:
                g.number_of_found_solutions += 1
                g.found_nonces.append(bytes(nonce))

        with g.lock:
            g.number_of_mining_iterations += 1


# ============================================================
# Pool generation (runs once at startup)
# ============================================================
def generate_pool(mining_seed: bytes) -> bytes:
    """
    Generate random2 pool from mining seed.
    The full pool is ~512MB. For initial CPU testing, we use a smaller pool.
    """
    from qubic_k12 import keccak_p1600_permute_12

    # Use a reasonable pool size for CPU (full 512MB would take too long)
    POOL_SIZE = 1024 * 1024 * 16  # 16MB for testing
    pool = bytearray(POOL_SIZE)
    state = bytearray(200)
    state[:32] = mining_seed[:32]

    for i in range(0, POOL_SIZE, 200):
        keccak_p1600_permute_12(state)
        end = min(i + 200, POOL_SIZE)
        pool[i:end] = state[:end - i]

    return bytes(pool)


# ============================================================
# Main — qli-Client Custom Runner Entry Point
# ============================================================
def main():
    # ---- Parse CLI ----
    # Custom runner mode: <IP> <ID> <THREADS>
    # Standalone mode:    <IP> <PORT> <ID> <SIGNING_SEED> <MINING_SEED> <THREADS>
    if len(sys.argv) == 4:
        # qli-Client custom runner mode
        node_ip = sys.argv[1]
        mining_id = sys.argv[2]
        num_threads = int(sys.argv[3])
        signing_seed = None
        # In custom runner mode, the mining seed is derived from the ID
        mining_seed_hex = None
    elif len(sys.argv) == 7:
        # Standalone Qiner-compatible mode
        node_ip = sys.argv[1]
        NODE_PORT = int(sys.argv[2])
        mining_id = sys.argv[3]
        signing_seed = sys.argv[4]
        mining_seed_hex = sys.argv[5]
        num_threads = int(sys.argv[6])
    else:
        print("Usage (qli-Client runner): python tpu_miner.py <IP> <ID> <THREADS>")
        print("Usage (standalone):        python tpu_miner.py <IP> <PORT> <ID> "
              "<SIGNING_SEED> <MINING_SEED> <THREADS>")
        sys.exit(1)

    # ---- Setup ----
    g.mining_id = mining_id
    g.computor_public_key = get_public_key_from_identity(mining_id)

    # Mining seed
    if mining_seed_hex:
        g.random_seed = bytes.fromhex(mining_seed_hex)
    else:
        # Derive from identity for custom runner mode
        g.random_seed = kangaroo_twelve(mining_id.encode(), 32)

    # Print startup info (Qiner-style)
    print(f"Qiner is launched. Connecting to {node_ip}:{NODE_PORT}")
    print(f"{num_threads} threads are used.")
    sys.stdout.flush()

    # Generate pool
    pool = generate_pool(g.random_seed)

    # ---- Start mining threads ----
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=mining_thread_proc, args=(pool,), daemon=True)
        t.start()
        threads.append(t)

    # ---- Main loop: stats output + solution submission ----
    # Matches Qiner.cpp main loop exactly
    prev_iterations = 0
    timestamp = time.time()

    while g.state == 0:
        # Check for solutions to send
        nonce_to_send = None
        with g.lock:
            if g.found_nonces:
                nonce_to_send = g.found_nonces[0]

        if nonce_to_send:
            if submit_solution(nonce_to_send, node_ip):
                with g.lock:
                    g.found_nonces.pop(0)

        time.sleep(1)

        # Print stats every ~1 second (matching Qiner.cpp printf format)
        now = time.time()
        delta_ms = (now - timestamp) * 1000
        if delta_ms >= 1000:
            with g.lock:
                current_iterations = g.number_of_mining_iterations
                solutions = g.number_of_found_solutions

            its = int((current_iterations - prev_iterations) * 1000 / delta_ms) if delta_ms > 0 else 0

            # UTC time (matching Qiner format)
            utc_now = datetime.now(timezone.utc)

            # This is the exact format qli-Client parses:
            # It looks for "X solutions" and "Y it/s"
            print(f"|   {utc_now.strftime('%Y-%m-%d %H:%M:%S')}   |   "
                  f"{its} it/s   |   "
                  f"{solutions} solutions   |   "
                  f"{mining_id[:10]}...   |")
            sys.stdout.flush()  # CRITICAL: qli-Client reads STDOUT

            prev_iterations = current_iterations
            timestamp = now

    # ---- Shutdown ----
    print("Shutting down...Press Ctrl+C again to force stop.")
    sys.stdout.flush()

    for t in threads:
        t.join(timeout=5)

    print("Qiner is shut down.")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
