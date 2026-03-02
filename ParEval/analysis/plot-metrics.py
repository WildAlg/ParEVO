import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(df: pd.DataFrame, metrics_to_plot: list, grouping_column: str, use_log_scale: bool, output_dir: str = 'plots'):
    """
    Generates bar charts for each specified metric.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    x_axis_label = 'Problem Name' if grouping_column == 'problem name' else 'Problem Type'
    title_suffix = 'Problem Name' if grouping_column == 'problem name' else 'Problem Type'
    x_tick_rotation = 90 if grouping_column == 'problem name' else 45

    for metric in metrics_to_plot:
        try:
            fig, ax = plt.subplots(figsize=(15, 9))

            if metric == 'parlay_omp_seq_ratio':
                plot_df = df.groupby(['problem name', 'source'], sort=False)[metric].mean().unstack()
                plot_df.plot(kind='bar', ax=ax)
                ax.set_title(f'Mean {metric} by Problem Name and Source')
                ax.set_ylabel(f'Mean {metric}')
                ax.tick_params(axis='x', rotation=90)
                
                # Clean legend for this specific metric
                ax.legend(title='Source', loc='best')

            else:
                # Group by problem type/name, execution model, and source
                grouped_df = df.groupby([grouping_column, 'execution model', 'source'], sort=False)[metric].mean()
                grouped_df = grouped_df.unstack(level=['execution model', 'source'])

                grouped_df.plot(kind='bar', ax=ax)

                ax.set_title(f'Mean {metric} by {title_suffix}, Execution Model, and Source')
                ax.set_xlabel(x_axis_label)
                ax.set_ylabel(f'Mean {metric}')
                ax.tick_params(axis='x', rotation=x_tick_rotation)

                if use_log_scale:
                    ax.set_yscale('log')
                    ax.set_ylabel(f'Mean {metric} (Log Scale)')

                # --- FIX: Clean up Legend Labels ---
                # Pandas defaults to labels like "('omp', 'source_name')". 
                # We rename them to "omp | source_name"
                new_labels = [f"{m} | {s}" for m, s in grouped_df.columns]
                ax.legend(new_labels, title='Execution Model | Source')
                # -----------------------------------

            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            log_suffix = '_log' if use_log_scale else ''
            filename = f'{metric.replace("@", "_")}_{grouping_column.replace(" ", "_")}{log_suffix}_bar_chart.pdf'
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, format='pdf', bbox_inches='tight')
            print(f"Plot saved to {output_path}")

            plt.close(fig)

        except KeyError:
            print(f"Warning: The metric '{metric}' or grouping column was not found. Skipping plot.")
        except Exception as e:
            print(f"An error occurred while plotting {metric}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze and plot performance data from multiple CSV files.')
    parser.add_argument('-i', '--input_files', 
                        type=str, 
                        nargs='+',
                        required=True,
                        help='Paths to the input CSV data files (space-separated).')
    
    # --- ADDED: Aliases argument ---
    parser.add_argument('-a', '--aliases',
                        type=str,
                        nargs='+',
                        help='Short names for the input files to use in legends (must match order of input files).')
    # -------------------------------

    parser.add_argument('-o', '--output_dir',
                        type=str,
                        default='plots',
                        help='Directory to save the generated plots. Default is "plots".')
    parser.add_argument('--problem-wise',
                        action='store_true',
                        help='Plot problem-wise metrics instead of problem-type-wise metrics.')
    parser.add_argument('--log-scale',
                        action='store_true',
                        help='Use a logarithmic scale for the y-axis.')
    
    args = parser.parse_args()

    # Validate aliases length
    if args.aliases and len(args.aliases) != len(args.input_files):
        print(f"Error: You provided {len(args.input_files)} input files but {len(args.aliases)} aliases.")
        print("Please provide exactly one alias per input file.")
        return

    dfs = []
    for idx, f in enumerate(args.input_files):
        try:
            df = pd.read_csv(f)
            
            # --- MODIFIED: Use alias if provided, else filename ---
            if args.aliases:
                source_name = args.aliases[idx]
            else:
                source_name = os.path.splitext(os.path.basename(f))[0]
            
            df['source'] = source_name
            # ------------------------------------------------------
            
            dfs.append(df)
            print(f"Successfully read data from: {f} (labeled as: {source_name})")
        except FileNotFoundError:
            print(f"Error: The file '{f}' was not found.")
            return

    if not dfs:
        print("No valid input files. Exiting.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Ensure source order is preserved for consistent coloring/legend
    if args.aliases:
        source_categories = args.aliases
    else:
        source_categories = [os.path.splitext(os.path.basename(f))[0] for f in args.input_files]
        
    df["source"] = pd.Categorical(df["source"], categories=source_categories, ordered=True)

    grouping_column = 'problem name' if args.problem_wise else 'problem type'

    metric_list = [
        'build@1', 'pass@1', 'speedup@1', 'speedup_max@1', 'efficiency@1', 'efficiency_max@1',
        'build@5', 'pass@5', 'speedup@5', 'speedup_max@5', 'efficiency@5', 'efficiency_max@5',
        'build@10', 'pass@10', 'speedup@10', 'speedup_max@10', 'efficiency@10', 'efficiency_max@10',
        'build@20', 'pass@20', 'speedup@20', 'speedup_max@20', 'efficiency@20', 'efficiency_max@20',
        'parlay_omp_seq_ratio'
    ]
    
    metrics_to_plot = [m for m in metric_list if m in df.columns]
    
    if not metrics_to_plot:
        print("No plottable metrics found in the CSVs. Exiting.")
        return

    plot_metrics(df, metrics_to_plot, grouping_column, args.log_scale, args.output_dir)

if __name__ == '__main__':
    main()
    
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os

# def plot_metrics(df: pd.DataFrame, metrics_to_plot: list, grouping_column: str, use_log_scale: bool, output_dir: str = 'plots'):
#     """
#     Generates bar charts for each specified metric, comparing different execution models
#     across either problem types or individual problems, for multiple CSV sources.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created output directory: {output_dir}")

#     x_axis_label = 'Problem Name' if grouping_column == 'problem name' else 'Problem Type'
#     title_suffix = 'Problem Name' if grouping_column == 'problem name' else 'Problem Type'
#     x_tick_rotation = 90 if grouping_column == 'problem name' else 45

#     for metric in metrics_to_plot:
#         try:
#             fig, ax = plt.subplots(figsize=(15, 9))

#             if metric == 'parlay_omp_seq_ratio':
#                 plot_df = df.groupby(['problem name', 'source'], sort=False)[metric].mean().unstack()
#                 plot_df.plot(kind='bar', ax=ax)
#                 ax.set_title(f'Mean {metric} by Problem Name and Source')
#                 ax.set_ylabel(f'Mean {metric}')
#                 ax.tick_params(axis='x', rotation=90)
#             else:
#                 # Group by problem type/name, execution model, and source
#                 grouped_df = df.groupby([grouping_column, 'execution model', 'source'], sort=False)[metric].mean()
#                 grouped_df = grouped_df.unstack(level=['execution model', 'source'])

#                 grouped_df.plot(kind='bar', ax=ax)

#                 ax.set_title(f'Mean {metric} by {title_suffix}, Execution Model, and Source')
#                 ax.set_xlabel(x_axis_label)
#                 ax.set_ylabel(f'Mean {metric}')
#                 ax.tick_params(axis='x', rotation=x_tick_rotation)

#                 if use_log_scale:
#                     ax.set_yscale('log')
#                     ax.set_ylabel(f'Mean {metric} (Log Scale)')

#                 ax.legend(title='Execution Model / Source')

#             ax.grid(axis='y', linestyle='--', alpha=0.7)
#             plt.tight_layout()

#             log_suffix = '_log' if use_log_scale else ''
#             filename = f'{metric.replace("@", "_")}_{grouping_column.replace(" ", "_")}{log_suffix}_bar_chart.pdf'
#             output_path = os.path.join(output_dir, filename)
#             fig.savefig(output_path, format='pdf', bbox_inches='tight')
#             # fig.savefig(output_path, dpi=300)
#             print(f"Plot saved to {output_path}")

#             plt.close(fig)

#         except KeyError:
#             print(f"Warning: The metric '{metric}' or grouping column was not found. Skipping plot.")
#         except Exception as e:
#             print(f"An error occurred while plotting {metric}: {e}")

# def main():
#     parser = argparse.ArgumentParser(description='Analyze and plot performance data from multiple CSV files.')
#     parser.add_argument('-i', '--input_files', 
#                         type=str, 
#                         nargs='+',
#                         required=True,
#                         help='Paths to the input CSV data files (space-separated).')
#     parser.add_argument('-o', '--output_dir',
#                         type=str,
#                         default='plots',
#                         help='Directory to save the generated plots. Default is "plots".')
#     parser.add_argument('--problem-wise',
#                         action='store_true',
#                         help='Plot problem-wise metrics instead of problem-type-wise metrics.')
#     parser.add_argument('--log-scale',
#                         action='store_true',
#                         help='Use a logarithmic scale for the y-axis.')
    
#     args = parser.parse_args()

#     dfs = []
#     for f in args.input_files:
#         try:
#             df = pd.read_csv(f)
#             df['source'] = os.path.splitext(os.path.basename(f))[0]  # tag source by filename
#             dfs.append(df)
#             print(f"Successfully read data from: {f}")
#         except FileNotFoundError:
#             print(f"Error: The file '{f}' was not found.")
#             return

#     if not dfs:
#         print("No valid input files. Exiting.")
#         return

#     df = pd.concat(dfs, ignore_index=True)

#     source_order = [os.path.splitext(os.path.basename(f))[0] for f in args.input_files]
#     df["source"] = pd.Categorical(df["source"], categories=source_order, ordered=True)

#     grouping_column = 'problem name' if args.problem_wise else 'problem type'

#     metric_list = [
#         'build@1', 'pass@1', 'speedup@1', 'speedup_max@1', 'efficiency@1', 'efficiency_max@1',
#         'build@5', 'pass@5', 'speedup@5', 'speedup_max@5', 'efficiency@5', 'efficiency_max@5',
#         'build@10', 'pass@10', 'speedup@10', 'speedup_max@10', 'efficiency@10', 'efficiency_max@10',
#         'build@20', 'pass@20', 'speedup@20', 'speedup_max@20', 'efficiency@20', 'efficiency_max@20',
#         'parlay_omp_seq_ratio'
#     ]
    
#     metrics_to_plot = [m for m in metric_list if m in df.columns]
    
#     if not metrics_to_plot:
#         print("No plottable metrics found in the CSVs. Exiting.")
#         return

#     plot_metrics(df, metrics_to_plot, grouping_column, args.log_scale, args.output_dir)

# if __name__ == '__main__':
#     main()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os

# def plot_metrics(df: pd.DataFrame, metrics_to_plot: list, grouping_column: str, use_log_scale: bool, output_dir: str = 'plots'):
#     """
#     Generates bar charts for each specified metric, comparing different execution models
#     across either problem types or individual problems.

#     Args:
#         df (pd.DataFrame): The input DataFrame containing the analysis data.
#         metrics_to_plot (list): A list of column names to plot.
#         grouping_column (str): The column to use for grouping the data.
#                                 Can be 'problem type' or 'problem name'.
#         use_log_scale (bool): Whether to use a logarithmic scale for the y-axis.
#         output_dir (str): The directory to save the plots.
#     """
#     # Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created output directory: {output_dir}")

#     # Determine the title and x-axis label based on the grouping column
#     x_axis_label = 'Problem Name' if grouping_column == 'problem name' else 'Problem Type'
#     title_suffix = 'Problem Name' if grouping_column == 'problem name' else 'Problem Type'
#     x_tick_rotation = 90 if grouping_column == 'problem name' else 45

#     # Generate a plot for each metric
#     for metric in metrics_to_plot:
#         try:
#             fig, ax = plt.subplots(figsize=(15, 9))

#             # Special case for the ratio metric, which is always plotted by problem name
#             if metric == 'parlay_omp_seq_ratio':
#                 plot_df = df.groupby('problem name')[metric].mean().sort_values(ascending=False)
#                 plot_df.plot(kind='bar', ax=ax, color='skyblue')
#                 ax.set_title(f'Mean {metric} by Problem Name')
#                 ax.set_ylabel(f'Mean {metric}')
#                 ax.tick_params(axis='x', rotation=90)
#             else:
#                 # Group data by the specified column and execution model, then calculate the mean
#                 grouped_df = df.groupby([grouping_column, 'execution model'])[metric].mean().unstack()
                
#                 # Plot the grouped data for the current metric
#                 grouped_df.plot(kind='bar', ax=ax)
            
#                 # Set the title and labels
#                 ax.set_title(f'Mean {metric} by {title_suffix} and Execution Model')
#                 ax.set_xlabel(x_axis_label)
#                 ax.set_ylabel(f'Mean {metric}')
#                 ax.tick_params(axis='x', rotation=x_tick_rotation)
                
#                 # Apply logarithmic scale if requested
#                 if use_log_scale:
#                     ax.set_yscale('log')
#                     # Also adjust the y-label to reflect the log scale
#                     ax.set_ylabel(f'Mean {metric} (Log Scale)')
                
#                 # Add a legend
#                 ax.legend(title='Execution Model')

#             # Add grid for better readability
#             ax.grid(axis='y', linestyle='--', alpha=0.7)

#             # Adjust layout to prevent labels from being cut off
#             plt.tight_layout()

#             # Save the plot to a file
#             log_suffix = '_log' if use_log_scale else ''
#             filename = f'{metric.replace("@", "_")}_{grouping_column.replace(" ", "_")}{log_suffix}_bar_chart.png'
#             output_path = os.path.join(output_dir, filename)
#             fig.savefig(output_path, dpi=300)
#             print(f"Plot saved to {output_path}")

#             # Close the figure to free memory
#             plt.close(fig)

#         except KeyError:
#             print(f"Warning: The metric '{metric}' or grouping column was not found. Skipping plot.")
#         except Exception as e:
#             print(f"An error occurred while plotting {metric}: {e}")

# def main():
#     """
#     Main function to read data and call the plotting function.
#     """
#     parser = argparse.ArgumentParser(description='Analyze and plot performance data from a CSV file.')
#     parser.add_argument('-i', '--input_file', 
#                         type=str, 
#                         required=True,
#                         help='Path to the input CSV data file.')
#     parser.add_argument('-o', '--output_dir',
#                         type=str,
#                         default='plots',
#                         help='Directory to save the generated plots. Default is "plots".')
#     parser.add_argument('--problem-wise',
#                         action='store_true',
#                         help='Plot problem-wise metrics instead of problem-type-wise metrics.')
#     parser.add_argument('--log-scale',
#                         action='store_true',
#                         help='Use a logarithmic scale for the y-axis.')
    
#     args = parser.parse_args()

#     try:
#         df = pd.read_csv(args.input_file)
#         print(f"Successfully read data from: {args.input_file}")
#     except FileNotFoundError:
#         print(f"Error: The file '{args.input_file}' was not found.")
#         return

#     # Determine the column to use for grouping
#     grouping_column = 'problem name' if args.problem_wise else 'problem type'

#     # List of all metrics to plot
#     metric_list = [
#         'build@1', 'pass@1', 'speedup@1', 'speedup_max@1', 'efficiency@1', 'efficiency_max@1',
#         'build@5', 'pass@5', 'speedup@5', 'speedup_max@5', 'efficiency@5', 'efficiency_max@5',
#         'build@10', 'pass@10', 'speedup@10', 'speedup_max@10', 'efficiency@10', 'efficiency_max@10',
#         'build@20', 'pass@20', 'speedup@20', 'speedup_max@20', 'efficiency@20', 'efficiency_max@20',
#         'parlay_omp_seq_ratio'
#     ]
    
#     # Check which metrics are actually in the DataFrame
#     metrics_to_plot = [m for m in metric_list if m in df.columns]
    
#     if not metrics_to_plot:
#         print("No plotable metrics found in the CSV. Exiting.")
#         return

#     plot_metrics(df, metrics_to_plot, grouping_column, args.log_scale, args.output_dir)

# if __name__ == '__main__':
#     main()

