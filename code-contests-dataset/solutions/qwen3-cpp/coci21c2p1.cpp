#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n;
    cin >> n;
    
    // Keep track of subsection counts for each section
    vector<int> subsection_counts(100001, 0);
    // Keep track of subsubsection counts for each section  
    vector<int> subsubsection_counts(100001, 0);
    
    int current_section = 0;
    
    for (int i = 0; i < n; i++) {
        string command, title;
        cin >> command >> title;
        
        if (command == "section") {
            current_section++;
            subsection_counts[current_section] = 0;
            subsubsection_counts[current_section] = 0;
            cout << current_section << " " << title << "\n";
        }
        else if (command == "subsection") {
            subsection_counts[current_section]++;
            subsubsection_counts[current_section] = 0;
            cout << current_section << "." << subsection_counts[current_section] << " " << title << "\n";
        }
        else if (command == "subsubsection") {
            subsubsection_counts[current_section]++;
            cout << current_section << "." << subsection_counts[current_section] << "." << subsubsection_counts[current_section] << " " << title << "\n";
        }
    }
    
    return 0;
}