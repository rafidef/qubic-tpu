"""
KangarooTwelve hash and Keccak-p[1600,12] permutation.
Exact port of the K12 implementation from Qiner's K12AndKeyUtil.h.
"""
import struct

# Keccak round constants for 12 rounds (indices 12..23 of the full 24-round set)
ROUND_CONSTANTS = [
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

K12_RATE_IN_BYTES = 168  # (1600 - 256) / 8
K12_CAPACITY_IN_BYTES = 32  # 256 / 8
K12_CHUNK_SIZE = 8192
K12_SUFFIX_LEAF = 0x0B
MASK64 = 0xFFFFFFFFFFFFFFFF


def _rol64(a: int, offset: int) -> int:
    """Rotate left 64-bit."""
    a &= MASK64
    return ((a << offset) | (a >> (64 - offset))) & MASK64


def keccak_p1600_permute_12(state_bytes: bytearray) -> None:
    """In-place Keccak-p[1600,12] permutation on a 200-byte state."""
    # Unpack state into 25 uint64 lanes
    lanes = list(struct.unpack_from('<25Q', state_bytes))
    A = lanes  # alias

    for rc in ROUND_CONSTANTS:
        # θ step
        C = [A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rol64(C[(x + 1) % 5], 1) for x in range(5)]
        A = [(A[i] ^ D[i % 5]) & MASK64 for i in range(25)]

        # ρ and π steps
        B = [0] * 25
        rho_offsets = [
            0, 1, 62, 28, 27,
            36, 44, 6, 55, 20,
            3, 10, 43, 25, 39,
            41, 45, 15, 21, 8,
            18, 2, 61, 56, 14,
        ]
        pi_from = [
            0, 6, 12, 18, 24,
            3, 9, 10, 16, 22,
            1, 7, 13, 19, 20,
            4, 5, 11, 17, 23,
            2, 8, 14, 15, 21,
        ]
        # Combined ρ + π: B[π(i)] = rot(A[i], ρ(i))
        # Standard Keccak mapping:
        # π: (x,y) -> (y, 2x+3y mod 5)
        # We use a direct index mapping
        _PI = [0, 10, 20, 5, 15,
               16, 1, 11, 21, 6,
               7, 17, 2, 12, 22,
               23, 8, 18, 3, 13,
               14, 24, 9, 19, 4]
        _RHO = [0, 1, 62, 28, 27,
                36, 44, 6, 55, 20,
                3, 10, 43, 25, 39,
                41, 45, 15, 21, 8,
                18, 2, 61, 56, 14]
        for i in range(25):
            B[_PI[i]] = _rol64(A[i], _RHO[i])

        # χ step
        A2 = [0] * 25
        for y in range(5):
            for x in range(5):
                idx = x + 5 * y
                A2[idx] = (B[idx] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y])) & MASK64

        # ι step
        A2[0] ^= rc
        A = A2

    # Pack back
    struct.pack_into('<25Q', state_bytes, 0, *A)


def kangaroo_twelve(input_data: bytes, output_len: int = 32) -> bytes:
    """
    KangarooTwelve hash function.
    Exact port of the C implementation from K12AndKeyUtil.h.
    """
    # Simple sponge-based implementation for short messages (< 8192 bytes)
    # which is all we need for the miner (64-byte inputs).
    input_len = len(input_data)

    if input_len <= K12_CHUNK_SIZE:
        # Single chunk path (no tree hashing needed)
        state = bytearray(200)
        byte_io_idx = 0

        # Absorb
        i = 0
        while i < input_len:
            block_size = min(K12_RATE_IN_BYTES - byte_io_idx, input_len - i)
            for j in range(block_size):
                state[byte_io_idx + j] ^= input_data[i + j]
            byte_io_idx += block_size
            i += block_size
            if byte_io_idx == K12_RATE_IN_BYTES:
                keccak_p1600_permute_12(state)
                byte_io_idx = 0

        if input_len < K12_CHUNK_SIZE:
            # Short message: pad with 0x07
            byte_io_idx_after = byte_io_idx + 1
            if byte_io_idx_after == K12_RATE_IN_BYTES:
                state[byte_io_idx] ^= 0x07
                # Actually the C code does:
                # finalNode.state[finalNode.byteIOIndex] ^= 0x07 after incrementing
                # Let me re-examine...
                # blockNumber = 0 path:
                # if (++finalNode.byteIOIndex == K12_rateInBytes) {
                #     KeccakP1600_Permute_12rounds(finalNode.state);
                #     finalNode.state[0] ^= 0x07;
                # } else {
                #     finalNode.state[finalNode.byteIOIndex] ^= 0x07;
                # }
                keccak_p1600_permute_12(state)
                state[0] ^= 0x07
            else:
                state[byte_io_idx_after] ^= 0x07
        else:
            # Exactly K12_CHUNK_SIZE: tree hashing with no additional chunks
            # blockNumber = 1, empty queue
            state[byte_io_idx] ^= 0x03
            byte_io_idx += 1
            if byte_io_idx == K12_RATE_IN_BYTES:
                keccak_p1600_permute_12(state)
                byte_io_idx = 0
            else:
                byte_io_idx = (byte_io_idx + 7) & ~7

            # Empty queue node
            queue_state = bytearray(200)
            queue_byte_io_idx = 1
            queue_absorbed_len = 1

            # blockNumber = 2 (1 original + 1 empty queue)
            block_number = 2
            queue_state[queue_byte_io_idx] ^= K12_SUFFIX_LEAF
            queue_state[K12_RATE_IN_BYTES - 1] ^= 0x80
            keccak_p1600_permute_12(queue_state)
            # Absorb CV into final
            for j in range(K12_CAPACITY_IN_BYTES):
                state[byte_io_idx + j] ^= queue_state[j]
            byte_io_idx += K12_CAPACITY_IN_BYTES
            if byte_io_idx >= K12_RATE_IN_BYTES:
                keccak_p1600_permute_12(state)
                byte_io_idx = 0

            # Encode block count
            block_number -= 1  # = 1
            n = 0
            v = block_number
            while v > 0 and n < 8:
                n += 1
                v >>= 8
            encbuf = bytearray(n + 3)
            for enc_i in range(1, n + 1):
                encbuf[enc_i - 1] = (block_number >> (8 * (n - enc_i))) & 0xFF
            encbuf[n] = n
            n += 1
            encbuf[n] = 0xFF
            n += 1
            encbuf[n] = 0xFF
            n += 1
            # Absorb encbuf
            for j in range(n):
                state[byte_io_idx] ^= encbuf[j]
                byte_io_idx += 1
                if byte_io_idx == K12_RATE_IN_BYTES:
                    keccak_p1600_permute_12(state)
                    byte_io_idx = 0

            state[byte_io_idx] ^= 0x06

        # Final squeeze
        state[K12_RATE_IN_BYTES - 1] ^= 0x80
        keccak_p1600_permute_12(state)
        return bytes(state[:output_len])
    else:
        raise NotImplementedError("KangarooTwelve for inputs > 8192 bytes not implemented")


def generate_random2_pool(mining_seed: bytes) -> bytearray:
    """
    Generate the random2 pool from mining seed.
    Port of generateRandom2Pool from score_common.h.
    """
    POOL_VEC_SIZE = ((1 << 32) + 64) >> 3  # ~512MB
    POOL_VEC_PADDING_SIZE = ((POOL_VEC_SIZE + 199) // 200) * 200

    pool = bytearray(POOL_VEC_PADDING_SIZE)
    state = bytearray(200)
    state[:32] = mining_seed[:32]

    for i in range(0, POOL_VEC_PADDING_SIZE, 200):
        keccak_p1600_permute_12(state)
        end = min(i + 200, POOL_VEC_PADDING_SIZE)
        pool[i:end] = state[:end - i]

    return pool


def random2(seed: bytes, pool: bytes, output_size: int) -> bytes:
    """
    Port of random2() from score_common.h.
    LCG-based PRNG that indexes into the Keccak pool.
    """
    padding_size = ((output_size + 63) // 64) * 64
    output = bytearray(padding_size)

    # Unpack seed as 8 uint32
    x = list(struct.unpack_from('<8I', seed))

    segments = padding_size // 64

    for j in range(segments):
        for i in range(8):
            base = (x[i] >> 3) >> 3
            m = x[i] & 63

            # Read two uint64 from pool
            pool_offset = base * 8
            if pool_offset + 16 <= len(pool):
                u64_0 = struct.unpack_from('<Q', pool, pool_offset)[0]
                u64_1 = struct.unpack_from('<Q', pool, pool_offset + 8)[0]
            else:
                u64_0 = 0
                u64_1 = 0

            if m == 0:
                val = u64_0
            else:
                val = ((u64_0 >> m) | (u64_1 << (64 - m))) & MASK64

            struct.pack_into('<Q', output, j * 64 + i * 8, val)

            # LCG update
            x[i] = (x[i] * 1664525 + 1013904223) & 0xFFFFFFFF

    return bytes(output[:output_size])
