import requests
from bs4 import BeautifulSoup

def fetch_dmoj_data(problem_id):
    # Define the URLs
    problem_url = f"https://dmoj.ca/problem/{problem_id}"
    editorial_url = f"https://dmoj.ca/problem/{problem_id}/editorial"

    # Use a user-agent to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # --- Fetch Problem Statement ---
        response_prob = requests.get(problem_url, headers=headers)
        response_prob.raise_for_status()  # Check for HTTP errors
        soup_prob = BeautifulSoup(response_prob.text, 'html.parser')
        
        # DMOJ problem text is typically in a div with class 'content-description'
        problem_element = soup_prob.select_one('.content-description')
        problem_text = problem_element.get_text(separator='\n', strip=True) if problem_element else "Problem content not found."

        # --- Fetch Editorial ---
        response_edit = requests.get(editorial_url, headers=headers)
        response_edit.raise_for_status()
        soup_edit = BeautifulSoup(response_edit.text, 'html.parser')
        
        # Editorials also usually reside in 'content-description'
        editorial_element = soup_edit.select_one('.content-description')
        editorial_text = editorial_element.get_text(separator='\n', strip=True) if editorial_element else "Editorial content not found."

        # --- Store in a String ---
        combined_string = f"=== PROBLEM STATEMENT ===\n\n{problem_text}\n\n" \
                          f"=== EDITORIAL ===\n\n{editorial_text}"
        
        return combined_string

    except requests.exceptions.RequestException as e:
        # print(f"An error occurred during fetching: {e}")
        return f"=== PROBLEM STATEMENT ===\n\n{problem_text}\n\n"

# Execute and print the result
if __name__ == "__main__":
    import sys
    final_content = fetch_dmoj_data(problem_id = sys.argv[1])

    print("Write a basic C++ solution to the following problem. This should be a basic implementation of the idea in the editorial.\n" + final_content)