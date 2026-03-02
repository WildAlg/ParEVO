import pandas as pd
import numpy as np
import argparse

def main(args):
    """
    Parses command-line arguments, reads a CSV file, performs data analysis,
    and saves the results to a new CSV file.
    """

    # Read the data from the specified input file
    try:
        df = pd.read_csv(args.input_file)
        print(f"Successfully read data from: {args.input_file}")
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
        return
    
    # Display the first few rows of the DataFrame
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head(5))

    # Create a dictionary to hold the aggregated data
    metrics = {
        'Metric': [],
        'Parlay': [],
        'OMP': []
    }

    # List of metrics to be processed
    metric_list = [
        'pass@1', 'speedup@1', 'speedup@5', 'speedup@10', 'speedup@20',
        'speedup_max@1', 'speedup_max@5', 'speedup_max@10', 'speedup_max@20',
        'efficiency@1', 'efficiency@5', 'efficiency@10', 'efficiency@20',
        'efficiency_max@1', 'efficiency_max@5', 'efficiency_max@10', 'efficiency_max@20'
    ]

    # Populate the dictionary with mean values for each metric
    for metric in metric_list:
        metrics['Metric'].append(metric)
        metrics['Parlay'].append(df[df["execution model"] == "parlay"][metric].mean())
        metrics['OMP'].append(df[df["execution model"] == "omp"][metric].mean())

    # Handle the special 'Parlay vs OMP Sequential Runtime Ratio' metric
    metrics['Metric'].append('Parlay vs OMP Sequential Runtime Ratio')
    metrics['Parlay'].append(df[df["execution model"] == "parlay"]["parlay_omp_seq_ratio"].mean())
    metrics['OMP'].append('N/A') # This metric is a ratio of the two, not a separate value for OMP

    # Create the DataFrame from the dictionary
    metrics_df = pd.DataFrame(metrics)
    
    # Print the final DataFrame
    print("\nAggregated Metrics Summary:")
    print(metrics_df)

    # Save the final DataFrame to a CSV file
    try:
        metrics_df.to_csv(args.output_file, index=False)
        print(f"\nSuccessfully saved the summary to: {args.output_file}")
    except IOError as e:
        print(f"Error saving file '{args.output_file}': {e}")


if __name__ == '__main__':
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description='Analyze Parlay and OMP performance data from a CSV file.')
    
    # Add the required argument for the input file path
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the input CSV data file.')

    # Add the optional argument for the output file path with a default value
    parser.add_argument('-o', '--output_file', type=str, default='metrics_summary.csv', help='Path to the output CSV file where the summary will be saved.')
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    main(args)
