import sys
import random

# Increase recursion depth for deep Segment Tree operations
sys.setrecursionlimit(300000)

# --- Hashing Configuration ---
MOD1 = 10**9 + 7
BASE1 = 313
MOD2 = 10**9 + 9
BASE2 = 317

# Global precomputed arrays
pow1 = []
pow2 = []
sum_pow1 = []
sum_pow2 = []

def precompute_hashes(n):
    """Precomputes powers and geometric series sums for hashing."""
    global pow1, pow2, sum_pow1, sum_pow2
    
    pow1 = [0] * (n + 2)
    pow2 = [0] * (n + 2)
    sum_pow1 = [0] * (n + 2)
    sum_pow2 = [0] * (n + 2)

    pow1[0] = 1
    pow2[0] = 1
    sum_pow1[0] = 0
    sum_pow2[0] = 0

    s1, s2 = 0, 0
    p1, p2 = 1, 1

    for i in range(n + 1):
        pow1[i] = p1
        pow2[i] = p2
        sum_pow1[i] = s1
        sum_pow2[i] = s2
        
        s1 = (s1 + p1) % MOD1
        s2 = (s2 + p2) % MOD2
        p1 = (p1 * BASE1) % MOD1
        p2 = (p2 * BASE2) % MOD2

def get_pow_sum(L, R):
    """
    Returns (sum(BASE1^k), sum(BASE2^k)) for k in [L, R].
    Used to calculate the hash of a range filled with a constant value.
    """
    v1 = (sum_pow1[R + 1] - sum_pow1[L] + MOD1) % MOD1
    v2 = (sum_pow2[R + 1] - sum_pow2[L] + MOD2) % MOD2
    return v1, v2

# --- Segment Tree for Hashing ---
class SegTree:
    def __init__(self, n, init_vec):
        self.n = n
        self.tree1 = [0] * (4 * n + 1)
        self.tree2 = [0] * (4 * n + 1)
        self.lazy = [-1] * (4 * n + 1) # -1 indicates no pending update
        self._build(1, 1, n, init_vec)

    def _push_up(self, node):
        self.tree1[node] = (self.tree1[2 * node] + self.tree1[2 * node + 1]) % MOD1
        self.tree2[node] = (self.tree2[2 * node] + self.tree2[2 * node + 1]) % MOD2

    def _push_down(self, node, L, R):
        if self.lazy[node] != -1:
            mid = (L + R) // 2
            val = self.lazy[node]

            # Left Child
            self.lazy[2 * node] = val
            p_left1, p_left2 = get_pow_sum(L, mid)
            self.tree1[2 * node] = (val * p_left1) % MOD1
            self.tree2[2 * node] = (val * p_left2) % MOD2

            # Right Child
            self.lazy[2 * node + 1] = val
            p_right1, p_right2 = get_pow_sum(mid + 1, R)
            self.tree1[2 * node + 1] = (val * p_right1) % MOD1
            self.tree2[2 * node + 1] = (val * p_right2) % MOD2

            self.lazy[node] = -1

    def _build(self, node, L, R, vec):
        if L == R:
            # vec is 0-indexed, tree is 1-based logic for L/R
            val = vec[L - 1]
            self.tree1[node] = (val * pow1[L]) % MOD1
            self.tree2[node] = (val * pow2[L]) % MOD2
            return
        
        mid = (L + R) // 2
        self._build(2 * node, L, mid, vec)
        self._build(2 * node + 1, mid + 1, R, vec)
        self._push_up(node)

    def update(self, node, L, R, ql, qr, val):
        if ql <= L and R <= qr:
            self.lazy[node] = val
            p1, p2 = get_pow_sum(L, R)
            self.tree1[node] = (val * p1) % MOD1
            self.tree2[node] = (val * p2) % MOD2
            return

        self._push_down(node, L, R)
        mid = (L + R) // 2
        if ql <= mid:
            self.update(2 * node, L, mid, ql, qr, val)
        if qr > mid:
            self.update(2 * node + 1, mid + 1, R, ql, qr, val)
        self._push_up(node)

    def query_root(self):
        """Returns the hash of the entire string (node 1)."""
        return self.tree1[1], self.tree2[1]

# --- Helper Functions ---
def int_to_char(v):
    if v == 0: return 'J'
    if v == 1: return 'O'
    if v == 2: return 'I'
    return 'J'

def char_to_int(c):
    if c == 'J': return 0
    if c == 'O': return 1
    if c == 'I': return 2
    return 0

# --- Main Generator ---
def generate_test_case(N, Q, in_file_name, out_file_name):
    print(f"Generating test case with N={N}, Q={Q}...")
    
    # 1. Random Generation setup
    # Using random.choices is faster than loop for large strings
    sA_ints = [random.randint(0, 2) for _ in range(N)]
    sB_ints = [random.randint(0, 2) for _ in range(N)]
    sC_ints = [random.randint(0, 2) for _ in range(N)]

    strA = "".join(int_to_char(x) for x in sA_ints)
    strB = "".join(int_to_char(x) for x in sB_ints)
    strC = "".join(int_to_char(x) for x in sC_ints)

    with open(in_file_name, "w") as fin:
        fin.write(f"{N}\n{strA}\n{strB}\n{strC}\n{Q}\n")
        
        # 2. Precompute hashes
        precompute_hashes(N)
        target_hashes = set() # Using a set for O(1) lookups

        # The valid sequences are linear combinations a*S_A + b*S_B + c*S_C
        # such that a + b + c = 1 (mod 3).
        print("Precomputing target hashes...")
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if (a + b + c) % 3 != 1:
                        continue
                    
                    h1, h2 = 0, 0
                    # Python optimization: calculate hash in linear pass
                    # This is O(N) but done 9 times, so O(N) total.
                    for i in range(N):
                        val = (a * sA_ints[i] + b * sB_ints[i] + c * sC_ints[i]) % 3
                        h1 = (h1 + val * pow1[i+1]) % MOD1
                        h2 = (h2 + val * pow2[i+1]) % MOD2
                    target_hashes.add((h1, h2))

        # 3. Generate T_0
        # Initialize T as a copy of A to ensure "Yes" at start
        t_ints = list(sA_ints)
        strT = strA
        fin.write(f"{strT}\n")

        # 4. Initialize Solver
        st = SegTree(N, t_ints)
        
        with open(out_file_name, "w") as fout:
            # Helper to check and write
            def check_and_output():
                cur_hash = st.query_root()
                if cur_hash in target_hashes:
                    fout.write("Yes\n")
                else:
                    fout.write("No\n")

            # Check initial state
            check_and_output()

            # 5. Process Queries
            print("Processing queries and updating Segment Tree...")
            for j in range(Q):
                if j % 10000 == 0 and j > 0:
                    print(f"  Processed {j}/{Q} queries...")

                L = random.randint(1, N)
                R = random.randint(1, N)
                if L > R: L, R = R, L
                
                c_val = random.randint(0, 2)
                char_c = int_to_char(c_val)

                fin.write(f"{L} {R} {char_c}\n")

                # Update Segment Tree (root is node 1, range 1 to N)
                # Query L, R are 1-based
                st.update(1, 1, N, L, R, c_val)
                check_and_output()

    print(f"Done! Files generated: {in_file_name}, {out_file_name}")

if __name__ == "__main__":
    # === CONFIGURATION ===
    # Set the problem size here. 
    # N = 200,000 and Q = 200,000, time complexity O(QlogN).
    N_PARAM = 20000
    Q_PARAM = 20000
    
    with open("init.yml", "a") as f:
        f.write("\n")
    
    for i in range(1, 11):
        INPUT_FILE = f"gen.{i}.in"
        OUTPUT_FILE = f"gen.{i}.out"
        with open("init.yml", "a") as f:
            f.write(f"""- {{in: gen.{i}.in, out: gen.{i}.out, points: 1}}\n""")
    
        generate_test_case(N_PARAM, Q_PARAM, INPUT_FILE, OUTPUT_FILE)