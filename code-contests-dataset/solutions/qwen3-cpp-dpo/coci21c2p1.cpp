#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n;
    cin >> n;
    
    // Track counters for each level
    vector<int> subsection_count(100001, 0); // Number of subsections per section
    vector<int> subsubsection_count(100001, 0); // Number of subsubsections per subsection
    
    int current_section = 0;
    
    for (int i = 0; i < n; i++) {
        string type, title;
        cin >> type >> title;
        
        if (type == "section") {
            current_section++;
            subsection_count[current_section] = 0;
            subsubsection_count[current_section] = 0;
            cout << current_section << " " << title << "\n";
        } else if (type == "subsection") {
            subsection_count[current_section]++;
            int current_subsection = subsection_count[current_section];
            subsubsection_count[current_section] = 0;
            cout << current_section << "." << current_subsection << " " << title << "\n";
        } else { // subsubsection
            subsubsection_count[current_section]++;
            int current_subsubsection = subsubsection_count[current_section];
            cout << current_section << "." << subsection_count[current_section] 
                 << "." << current_subsubsection << " " << title << "\n";
        }
    }
    
    return 0;
}