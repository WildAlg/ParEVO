#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

// Map characters to integers: J=0, O=1, I=2
int charToInt(char c) {
    if (c == 'J') return 0;
    if (c == 'O') return 1;
    if (c == 'I') return 2;
    return -1;
}

// Map integers back to characters
char intToChar(int v) {
    if (v == 0) return 'J';
    if (v == 1) return 'O';
    if (v == 2) return 'I';
    return '?';
}

int N, Q;
string S[3]; // Initial strings SA, SB, SC
string T;    // The target string (mutable)
vector<string> patterns; // The 9 precalculated valid patterns

// Segment Tree Node
struct Node {
    // Bitmask of length 9. Bit k is 1 if T matches pattern k in this node's range.
    int match_mask; 
    
    // Precomputed masks for lazy updates. 
    // static_masks[c] (where c is 0, 1, or 2) tells us which patterns 
    // would match if T was filled entirely with character c in this range.
    int static_masks[3]; 
    
    // Lazy tag: -1 (none), 0 (set to J), 1 (set to O), 2 (set to I)
    int lazy; 
} tree[800005]; // Size ~ 4 * N

// Generate the 9 valid linear combinations
void generatePatterns() {
    int coeff[3];
    // Iterate all combinations of coefficients (a, b, c) in Z3
    for (coeff[0] = 0; coeff[0] < 3; coeff[0]++) {
        for (coeff[1] = 0; coeff[1] < 3; coeff[1]++) {
            for (coeff[2] = 0; coeff[2] < 3; coeff[2]++) {
                // Condition: a + b + c = 1 (mod 3)
                if ((coeff[0] + coeff[1] + coeff[2]) % 3 == 1) {
                    string pat = "";
                    for (int i = 0; i < N; i++) {
                        int val = 0;
                        for (int k = 0; k < 3; k++) {
                            val += coeff[k] * charToInt(S[k][i]);
                        }
                        val %= 3;
                        pat += intToChar(val);
                    }
                    patterns.push_back(pat);
                }
            }
        }
    }
}

// Push lazy tags down to children
void push(int node, int start, int end) {
    if (tree[node].lazy != -1) {
        int val = tree[node].lazy;
        
        // Apply to left child
        tree[2 * node].lazy = val;
        tree[2 * node].match_mask = tree[2 * node].static_masks[val];
        
        // Apply to right child
        tree[2 * node + 1].lazy = val;
        tree[2 * node + 1].match_mask = tree[2 * node + 1].static_masks[val];
        
        // Clear tag
        tree[node].lazy = -1;
    }
}

// Combine results from children
void pull(int node) {
    // Current range matches pattern k only if BOTH left and right children match pattern k
    tree[node].match_mask = tree[2 * node].match_mask & tree[2 * node + 1].match_mask;
}

// Build the Segment Tree
void build(int node, int start, int end) {
    tree[node].lazy = -1;
    
    if (start == end) {
        // LEAF NODE
        // 1. Calculate static_masks for J, O, I
        for (int c = 0; c < 3; c++) {
            int mask = 0;
            char char_c = intToChar(c);
            for (int k = 0; k < 9; k++) {
                // If pattern k has char c at this position, it's a match
                if (patterns[k][start] == char_c) {
                    mask |= (1 << k);
                }
            }
            tree[node].static_masks[c] = mask;
        }
        
        // 2. Initialize match_mask based on initial string T
        int current_val = charToInt(T[start]);
        tree[node].match_mask = tree[node].static_masks[current_val];
        return;
    }

    int mid = (start + end) / 2;
    build(2 * node, start, mid);
    build(2 * node + 1, mid + 1, end);

    // Combine static_masks from children (AND logic)
    for (int c = 0; c < 3; c++) {
        tree[node].static_masks[c] = tree[2 * node].static_masks[c] & tree[2 * node + 1].static_masks[c];
    }
    pull(node);
}

// Range Update: set T[l...r] to val
void update(int node, int start, int end, int l, int r, int val) {
    if (l > end || r < start) return;
    
    if (l <= start && end <= r) {
        // Range completely covered
        tree[node].lazy = val;
        tree[node].match_mask = tree[node].static_masks[val];
        return;
    }

    push(node, start, end);
    int mid = (start + end) / 2;
    update(2 * node, start, mid, l, r, val);
    update(2 * node + 1, mid + 1, end, l, r, val);
    pull(node);
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N)) return 0;
    cin >> S[0] >> S[1] >> S[2];
    cin >> Q;
    cin >> T;

    // 1. Precalculate the 9 valid patterns
    generatePatterns();
    
    // 2. Build the Segment Tree
    // Using 0-based indexing for the tree range
    build(1, 0, N - 1);

    // 3. Check initial T
    if (tree[1].match_mask > 0) cout << "Yes\n";
    else cout << "No\n";

    // 4. Process Queries
    for (int j = 0; j < Q; j++) {
        int l, r;
        char c;
        cin >> l >> r >> c;
        
        // Input is 1-based, convert to 0-based
        update(1, 0, N - 1, l - 1, r - 1, charToInt(c));
        
        // Check if T matches any valid pattern
        if (tree[1].match_mask > 0) cout << "Yes\n";
        else cout << "No\n";
    }

    return 0;
}