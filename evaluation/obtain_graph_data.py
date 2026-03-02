def convert_csr_to_adjacency_graph(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = list(map(int, f.readlines()[1:]))

    n = lines[0]
    m = lines[1]
    offsets = lines[2:2 + n]
    edges = lines[2 + n:]

    # Step 1: Compute degrees
    degrees = []
    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < n else m
        degrees.append(end - start)

    # Step 2: Sort neighbors and delta encode edges
    delta_encoded_edges = []
    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < n else m
        neighbors = sorted(edges[start:end])
        for j, neighbor in enumerate(neighbors):
            if j == 0:
                delta_encoded_edges.append(neighbor)
            else:
                delta_encoded_edges.append(neighbor - neighbors[j - 1])

    # Step 3: Write output (1 number per line)
    with open(output_file, 'w') as f:
        f.write(f"{n}\n")
        f.write(f"{m}\n")
        for d in degrees:
            f.write(f"{d}\n")
        for e in delta_encoded_edges:
            f.write(f"{e}\n")



# Example usage
# convert_csr_to_adjacency_graph("csr.txt", "adj_graph.txt")

# ...existing code...

if __name__ == "__main__":
    import argparse
    import sys
    import json
    import subprocess
    import os

    parser = argparse.ArgumentParser(description="Download graph data online and convert it to adjacency-graph format that ParlayLib accepts.")
    parser.add_argument("graph_data", type=str, help="Name of the graph data", nargs='?', default="brain_batch")
    parser.add_argument("--obtain_all", action="store_true", help="Download and convert all graph data in the JSON file")
    args = parser.parse_args()

    with open("graph_data_link.json", 'r') as f:
        graph_data_link_dict = json.load(f)

    def process_graph(graph_name):
        graph_data_link = graph_data_link_dict.get(graph_name)
        if graph_data_link is None:
            print(f"Graph data link for {graph_name} not found.")
            return

        print(f"Graph data link: {graph_data_link}")

        # Download the graph data, if it is not already downloaded 
        if not os.path.isdir("graph_data"):
            os.mkdir("graph_data")

        graph_data_path = f"graph_data/{graph_name}"
        if os.path.isfile(graph_data_path) is False:
            subprocess.run(["wget", graph_data_link], cwd="graph_data")

        # Strip the first and last column of the graph data downloaded 
        stripped_path = f"graph_data/{graph_name}_uv.txt"
        if os.path.isfile(stripped_path) is False:
            subprocess.run(
                f"awk '{{print $2, $3}}' graph_data/{graph_name} > graph_data/{graph_name}_uv.txt",
                shell=True
            )

        # Convert the graph data to CSR format 
        input_path = os.path.abspath(f"graph_data/{graph_name}_uv.txt")
        output_path = os.path.abspath(f"graph_data/{graph_name}_CSR.txt")
        if os.path.isfile(output_path) is False:
            subprocess.run(
                f"bazel run //utils:snap_converter -- -s -i {input_path} -o {output_path}", 
                shell=True,
                cwd="../gbbs"
            )

        # Convert the CSR format to adjacency-graph format that ParlayLib accepts 
        adj_path = f"graph_data/{graph_name}_adj.txt"
        if os.path.isfile(adj_path) is False:
            convert_csr_to_adjacency_graph(output_path, adj_path)

        print(f"Successfully downloaded {graph_name} and converted it to graph_data/{graph_name}_adj.txt")

    if args.obtain_all:
        for graph_name in graph_data_link_dict.keys():
            process_graph(graph_name)
    else:
        process_graph(args.graph_data)


# if __name__ == "__main__":
#     import argparse
#     import sys
#     import json
#     import subprocess
#     import os

#     # Argument parsing
#     parser = argparse.ArgumentParser(description="Download graph data online and convert it to adjacency-graph format that ParlayLib accepts.")
#     parser.add_argument("graph_data", type=str, help="Name of the graph data")
#     parser.add_argument("--obtain_all", action='store_true', help="Obtain all graph data available in the graph_data_link.json file")
#     args = parser.parse_args()
    
#     # Load graph data link
#     graph_data_link = None
#     with open("graph_data_link.json", 'r') as f:
#         graph_data_link_dict = json.load(f)
#         graph_data_link = graph_data_link_dict[args.graph_data]
    
#     if graph_data_link is None:
#         print(f"Graph data link for {args.graph_data} not found.")
#         sys.exit(1)
    
#     print(f"Graph data link: {graph_data_link}")

#     # Download the graph data, if it is not already downloaded 
#     graph_data_path = f"graph_data/{args.graph_data}"
#     if os.path.isfile(graph_data_path) is False:
#         subprocess.run(["wget", graph_data_link], cwd="graph_data")

#     # Strip the first and last column of the graph data downloaded 
#     # Check if the stripped version (_uv.txt) exists
#     stripped_path = f"graph_data/{args.graph_data}_uv.txt"
#     if os.path.isfile(stripped_path) is False:
#         subprocess.run(
#             f"awk '{{print $2, $3}}' graph_data/{args.graph_data} > graph_data/{args.graph_data}_uv.txt",
#             shell=True
#         )

#     # Convert the graph data to CSR format 
#     input_path = os.path.abspath(f"graph_data/{args.graph_data}_uv.txt")
#     output_path = os.path.abspath(f"graph_data/{args.graph_data}_CSR.txt")
#     # Check if the CSR (_CSR.txt) format has already been converted to
#     if os.path.isfile(output_path) is False:
#         subprocess.run(
#             f"bazel run //utils:snap_converter -- -s -i {input_path} -o {output_path}", 
#             shell=True,
#             cwd="../gbbs"
#         )

#     # Convert the CSR format to adjacency-graph format that ParlayLib accepts 
#     # Check if the adjacency-graph format has already been converted to
#     adj_path = f"graph_data/{args.graph_data}_adj.txt"
#     if os.path.isfile(adj_path) is False:
#         convert_csr_to_adjacency_graph(output_path, adj_path)
    
#     print(f"Successfully downloaded {args.graph_data} and converted it to graph_data/{args.graph_data}_adj.txt")
