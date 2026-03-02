"""
Evolution Results Viewer

Usage:
    python viewer.py                    # Start web server (default problem)
    python viewer.py ioi19p2            # Start web server with specific problem
    python viewer.py --export           # Export all problems to standalone bundle
    python viewer.py --export ioi19p2   # Export specific problem to standalone bundle
"""

from flask import Flask, render_template, jsonify
import csv
import json
import re
from pathlib import Path
import argparse
from config import RESULTS_DIR, LLM_PROVIDER, LANGUAGE

app = Flask(__name__)

# =============================================================================
# Data Loading
# =============================================================================

def get_available_problems():
    """Get list of available problem IDs from results directory"""
    if not RESULTS_DIR.exists():
        return []
    return sorted([d.name for d in RESULTS_DIR.iterdir() 
                   if d.is_dir() and (d / "evaluation_results.csv").exists()])


def load_results(problem_id):
    """Load results from CSV file for a specific problem"""
    results = []
    csv_file = RESULTS_DIR / problem_id / "evaluation_results.csv"
    
    if not csv_file.exists():
        return results
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['code'] = str(row['code'])
            row['iteration'] = int(row['iteration'])
            row['combined_score'] = float(row['combined_score'])
            row['tests_passed'] = int(row['tests_passed'])
            row['tests_total'] = int(row['tests_total'])
            row['avg_time'] = float(row['avg_time'])
            row['variance_time'] = float(row['variance_time'])
            row['max_time'] = float(row['max_time'])
            results.append(row)
    
    return results


# =============================================================================
# Flask Routes (for live server mode)
# =============================================================================

@app.route('/')
def index():
    problems = get_available_problems()
    default_problem = app.config.get('default_problem', problems[0] if problems else "")
    return render_template('index.html', problems=problems, default_problem=default_problem)


@app.route('/api/results/<problem_id>')
def get_results(problem_id):
    results = load_results(problem_id)
    return jsonify(results)


@app.route('/api/problems')
def get_problems():
    return jsonify(get_available_problems())


# =============================================================================
# Export functionality
# =============================================================================

def export_bundle(problem_ids=None):
    """Export standalone HTML viewer with embedded data"""
    
    # Get available problems
    all_problems = get_available_problems()
    
    if not all_problems:
        print("Error: No problems found in results directory")
        print(f"Results directory: {RESULTS_DIR}")
        return
    
    # Filter to specific problems if requested
    if problem_ids:
        problems_to_export = [p for p in problem_ids if p in all_problems]
        missing = [p for p in problem_ids if p not in all_problems]
        if missing:
            print(f"Warning: Problems not found: {', '.join(missing)}")
        if not problems_to_export:
            print("Error: No valid problems to export")
            return
    else:
        problems_to_export = all_problems
    
    # Create export directory
    export_dir = Path(f"export_{LLM_PROVIDER}_{LANGUAGE}")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to: {export_dir}")
    print(f"Problems: {', '.join(problems_to_export)}")
    
    # Load all results into a dict
    all_data = {}
    total_iterations = 0
    for problem_id in problems_to_export:
        results = load_results(problem_id)
        all_data[problem_id] = results
        total_iterations += len(results)
        print(f"  Loaded: {problem_id} ({len(results)} iterations)")
    
    # Read the template
    template_path = Path(__file__).parent / "templates" / "index.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Generate static options for the select
    default_problem = problems_to_export[0]
    options_html = '\n'.join([
        f'                <option value="{p}"{" selected" if p == default_problem else ""}>{p}</option>'
        for p in problems_to_export
    ])
    
    # Replace Jinja for loop with static options
    jinja_pattern = r'\{%\s*for\s+problem\s+in\s+problems\s*%\}.*?\{%\s*endfor\s*%\}'
    html_content = re.sub(jinja_pattern, options_html, html_content, flags=re.DOTALL)
    
    # Replace {{ default_problem }} with actual value
    html_content = html_content.replace('{{ default_problem }}', default_problem)
    
    # Generate embedded data script
    embedded_data = f"const EMBEDDED_DATA = {json.dumps(all_data)};"
    
    # Replace the loadResults function to use embedded data
    new_load_results = f'''async function loadResults() {{
            // EMBEDDED_DATA_MARKER
            results = EMBEDDED_DATA[currentProblemId] || [];
            renderIterationList();'''
    
    html_content = html_content.replace(
        "const response = await fetch(`/api/results/${currentProblemId}`); // RESULTS_ENDPOINT\n            results = await response.json();",
        "results = EMBEDDED_DATA[currentProblemId] || [];"
    )
    
    # Insert embedded data before the loadResults function
    html_content = html_content.replace(
        "// EMBEDDED_DATA_MARKER",
        ""
    )
    html_content = html_content.replace(
        "async function loadResults() {",
        f"{embedded_data}\n\n        async function loadResults() {{"
    )
    
    # Remove AUTO_REFRESH line
    html_content = re.sub(
        r"setInterval\(loadResults,\s*\d+\);\s*// AUTO_REFRESH.*\n?",
        "",
        html_content
    )
    
    # Write the exported HTML
    output_file = export_dir / "index.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Calculate file size
    file_size = output_file.stat().st_size
    size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / 1024 / 1024:.1f} MB"
    
    print(f"\nExport complete!")
    print(f"Output: {output_file} ({size_str})")
    print(f"Total: {len(problems_to_export)} problems, {total_iterations} iterations")
    print(f"\nJust double-click index.html to open in browser.")


# =============================================================================
# Main
# =============================================================================

def run_server(default_problem=None):
    """Run the Flask development server"""
    problems = get_available_problems()
    if default_problem and default_problem in problems:
        app.config['default_problem'] = default_problem
    elif problems:
        app.config['default_problem'] = problems[0]
    else:
        app.config['default_problem'] = ""
    
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Open browser at: http://localhost:5000")
    app.run(debug=True, port=5000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolution Results Viewer')
    parser.add_argument('--export', nargs='*', metavar='PROBLEM_ID',
                        help='Export to standalone bundle. Optionally specify problem IDs.')
    parser.add_argument('problem', nargs='?', default=None,
                        help='Default problem ID for server mode')
    
    args = parser.parse_args()
    
    if args.export is not None:
        # Export mode
        problem_ids = args.export if args.export else None
        export_bundle(problem_ids)
    else:
        # Server mode
        run_server(args.problem)