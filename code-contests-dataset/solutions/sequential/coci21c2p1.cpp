#include <iostream>
#include <string>

using namespace std;

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // Counters for the three levels
    int s0 = 0; // Section
    int s1 = 0; // Subsection
    int s2 = 0; // Subsubsection

    string command, title;

    for (int i = 0; i < n; i++) {
        cin >> command >> title;

        if (command == "section") {
            s0++;
            s1 = 0; // Reset subsection counter
            s2 = 0; // Reset subsubsection counter
            cout << s0 << " " << title << "\n";
        } 
        else if (command == "subsection") {
            s1++;
            s2 = 0; // Reset subsubsection counter
            cout << s0 << "." << s1 << " " << title << "\n";
        } 
        else if (command == "subsubsection") {
            s2++;
            cout << s0 << "." << s1 << "." << s2 << " " << title << "\n";
        }
    }

    return 0;
}