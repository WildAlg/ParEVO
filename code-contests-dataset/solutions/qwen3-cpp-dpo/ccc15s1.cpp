#include <iostream>
#include <vector>
#include <stack>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int K;
    cin >> K;
    
    stack<int> numbers;
    
    for (int i = 0; i < K; i++) {
        int num;
        cin >> num;
        
        if (num == 0) {
            if (!numbers.empty()) {
                numbers.pop();
            }
        } else {
            numbers.push(num);
        }
    }
    
    long long sum = 0;
    while (!numbers.empty()) {
        sum += numbers.top();
        numbers.pop();
    }
    
    cout << sum << "\n";
    
    return 0;
}