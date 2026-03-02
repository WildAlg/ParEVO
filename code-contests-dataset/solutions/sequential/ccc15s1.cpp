#include <iostream>
#include <stack>

using namespace std;

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int k;
    if (!(cin >> k)) return 0;

    stack<int> numbers;

    for (int i = 0; i < k; i++) {
        int num;
        cin >> num;

        if (num == 0) {
            // Boss made a mistake: remove the last number (Pop)
            // The problem guarantees we won't pop from an empty stack, 
            // but checking !empty() is good practice.
            if (!numbers.empty()) {
                numbers.pop();
            }
        } else {
            // Valid number: add to the list (Push)
            numbers.push(num);
        }
    }

    // Sum up all remaining numbers in the stack
    int total_sum = 0;
    while (!numbers.empty()) {
        total_sum += numbers.top();
        numbers.pop();
    }

    cout << total_sum << "\n";

    return 0;
}