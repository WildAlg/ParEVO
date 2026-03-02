#include <iostream>
#include <vector>
using namespace std;

int main() {
    int K;
    cin >> K;
    
    vector<int> numbers;
    
    for (int i = 0; i < K; i++) {
        int num;
        cin >> num;
        
        if (num == 0) {
            // Remove the last number if exists
            if (!numbers.empty()) {
                numbers.pop_back();
            }
        } else {
            // Add the number to our list
            numbers.push_back(num);
        }
    }
    
    // Calculate the sum
    long long sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    
    cout << sum << endl;
    
    return 0;
}