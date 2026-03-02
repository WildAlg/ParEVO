#include <iostream>
#include <vector>
using namespace std;

int main() {
    // Define calorie values for each menu item
    vector<int> burger_calories = {461, 431, 420, 0};     // 1-4 burger choices
    vector<int> side_calories = {100, 57, 70, 0};         // 1-4 side choices
    vector<int> drink_calories = {130, 160, 118, 0};      // 1-4 drink choices
    vector<int> dessert_calories = {167, 266, 75, 0};     // 1-4 dessert choices
    
    int burger_choice, side_choice, drink_choice, dessert_choice;
    
    // Read choices
    cin >> burger_choice >> side_choice >> drink_choice >> dessert_choice;
    
    // Calculate total calories
    int total_calories = burger_calories[burger_choice - 1] + 
                        side_calories[side_choice - 1] + 
                        drink_calories[drink_choice - 1] + 
                        dessert_calories[dessert_choice - 1];
    
    cout << "Your total Calorie count is " << total_calories << "." << endl;
    
    return 0;
}