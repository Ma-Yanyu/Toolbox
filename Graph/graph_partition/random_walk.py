# based on the setting
# either output experiments stats or sampled edges.

import pandas as pd
import csv
import argparse
from tqdm import tqdm
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def random_walk_uni(G, start_nodes, restart_prob=0.15, target_edge_count=1_000):
    
    start_nodes_in_graph = [node for node in start_nodes if node in G]
    
    # Make sure there are valid start nodes
    if not start_nodes_in_graph:
        raise ValueError("No valid start nodes found in the graph.")
    
    current_node = random.choice(start_nodes_in_graph)  # Start from a random valid disease/gene node
    sampled_edges = set()  # To store the sampled edges
    sampled_nodes = set([current_node])  # To store the sampled nodes

    edge_relation_map = nx.get_edge_attributes(G, 'relation')

    with tqdm(total=target_edge_count, desc="Sampling edges", unit="edge") as pbar:
        while len(sampled_edges) < target_edge_count:
            # With restart probability, jump back to one of the start nodes
            if random.random() < restart_prob:
                current_node = random.choice(start_nodes_in_graph)  # Ensure we choose from valid nodes
            else:
                # Get all outgoing neighbors and corresponding edges
                neighbors = list(G.neighbors(current_node))
                if neighbors:
                    # Group neighbors by relation type
                    edges_by_relation = {}
                    for next_node in neighbors:
                        relation = edge_relation_map.get((current_node, next_node))
                        if relation not in edges_by_relation:
                            edges_by_relation[relation] = []
                        edges_by_relation[relation].append((current_node, next_node))
                    
                    # Uniformly choose a relation type from the available ones
                    available_relations = list(edges_by_relation.keys())
                    chosen_relation = random.choice(available_relations)
                    
                    # Uniformly choose an edge that corresponds to the chosen relation type
                    u, v = random.choice(edges_by_relation[chosen_relation])
                    
                    # Check if the edge is already sampled, if not, add it to sampled edges
                    if (u, v) not in sampled_edges:
                        sampled_edges.add((u, v))  # Add the edge
                        pbar.update(1)  # Update progress bar
                    current_node = v  # Move to the next node
    
    return sampled_edges

def get_graph_attributes(G_sub, df):
    results = {}

    # Entropy of _type distribution
    type_counts = df['_type'].value_counts()
    type_probs = type_counts / type_counts.sum()
    entropy = -np.sum(type_probs * np.log2(type_probs))
    results['entropy'] = entropy

    # Average in-degree and out-degree
    avg_out_degree = sum(dict(G_sub.out_degree()).values()) / G_sub.number_of_nodes()
    results['avg_out_degree'] = avg_out_degree

    # Graph diameter (only if connected)
    # if nx.is_connected(G_sub.to_undirected()):
    #     diameter = nx.diameter(G_sub.to_undirected())
    #     results['diameter'] = diameter
    # else:
    #     results['diameter'] = None

    # Graph density
    density = nx.density(G_sub)
    results['density'] = density

    # Average clustering coefficient
    # clustering_coeff = nx.average_clustering(G_sub.to_undirected())
    # results['clustering_coeff'] = clustering_coeff

    # Number of nodes and edges
    results['num_nodes'] = G_sub.number_of_nodes()
    results['num_edges'] = G_sub.number_of_edges()

    return results

# def main(args):
#     df = pd.read_csv('/data/home/yyma/KGC-project/CM-BKG/data/raw/extracted_data_entity_text_yz.csv')
#     tri_path = '/data/home/yyma/KGC-project/CM-BKG/data/remove_assay/train.csv'
#     tri_df = pd.read_csv(tri_path)

#     id2type = defaultdict()
#     # create a node-to-type mapping dict
#     for index,row in tqdm(df.iterrows(), total=len(df)):
#         row_dict = row.to_dict()
#         id2type[str(int(row_dict['_id']))] = row_dict['_labels'][1:]

#     # filter the edges to 30 relation types
#     if args.clean_rels:
#         tri_rels = tri_df['_type'].value_counts()
#         top30_types = tri_rels.nlargest(30).index
#         filtered_df = tri_df[tri_df['_type'].isin(top30_types)]
#         filtered_df['_start'] = filtered_df['_start'].apply(lambda x: str(int(x)))
#         filtered_df['_end'] = filtered_df['_end'].apply(lambda x: str(int(x)))
#     else:
#         filtered_df = tri_df
#         filtered_df['_start'] = filtered_df['_start'].apply(lambda x: str(int(x)))
#         filtered_df['_end'] = filtered_df['_end'].apply(lambda x: str(int(x)))

#     # construct the graph
#     print(f"Constructing the knowledge graph as networkx object..")
#     G = nx.DiGraph()
#     for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
#         G.add_edge(row['_start'], row['_end'], relation=row['_type'])

#     print(f"Graph construction finished, starting to get graph attributes. ")
#     all_results = []
#     graph_attributes = get_graph_attributes(G, filtered_df)
#     graph_attributes['start_size'] = 0
#     graph_attributes['sample_ratio'] = 0
#     all_results.append(graph_attributes)

#     # intialize the starting nodes    
#     start_nodes = [k for k,v in id2type.items() if v in args.target_types and k in G]
    
#     print(f"The size of start nodes is {len(start_nodes)}")
#     if isinstance(args.start_size, list):
#         for start_size in args.start_size:
#             sampled_start_nodes = random.sample(start_nodes, start_size)
    
#             sample_size = int(len(filtered_df) * args.sample_ratio)
#             print(f"Starting random walk, target edge size is {sample_size}")

#             sampled_edges = random_walk_uni(G, sampled_start_nodes, target_edge_count=sample_size)

#             sampled_results = []
#             for _, row in filtered_df.iterrows():
#                 if (row['_start'], row['_end']) in sampled_edges:
#                     sampled_results.append(row)

#             sampled_df = pd.DataFrame(sampled_results)

#             G_sub = nx.DiGraph()
#             for _, row in sampled_df.iterrows():
#                 G_sub.add_edge(row['_start'], row['_end'], relation=row['_type'])

#             # Get graph attributes and add experiment-specific info (start_size and sample_ratio)
#             graph_attributes = get_graph_attributes(G_sub, sampled_df)
#             graph_attributes['start_size'] = start_size
#             graph_attributes['sample_ratio'] = args.sample_ratio

#             # Append the result to all_results
#             all_results.append(graph_attributes)

#         if args.save_to_csv:
#             output_df = pd.DataFrame(all_results)
#             output_df.to_csv(args.out_stat_path, index=False)
#             print(f"Results saved to {args.out_stat_path}")
#     else:
#         sampled_start_nodes = random.sample(start_nodes, args.start_size)
#         sample_size = int(len(filtered_df) * args.sample_ratio)
#         print(f"Starting random walk, target edge size is {sample_size}")

#         sampled_edges = random_walk_uni(G, sampled_start_nodes, target_edge_count=sample_size)

#         sampled_results = []
#         for _, row in filtered_df.iterrows():
#             if (row['_start'], row['_end']) in sampled_edges:
#                 sampled_results.append(row)

#         sampled_df = pd.DataFrame(sampled_results)
#         subgraph_edges = sampled_df[['_start', '_type', '_end']]
#         print(f'Saving sampled edges to {args.out_path}')

#         # Save to CSV (without index)
#         subgraph_edges.to_csv(args.out_path, index=False, header=False, sep='\t')

def main(args):
    df = pd.read_csv('/dataStor/home/yyma/KGC-project/CM-BKG/data/raw/extracted_data_entity_text_yz.csv')
    
    tri_df = pd.read_csv(args.tri_path)
    tri_df['_start'] = tri_df['_start'].apply(lambda x: str(int(x)))
    tri_df['_end'] = tri_df['_end'].apply(lambda x: str(int(x)))

    id2type = defaultdict()
    # create a node-to-type mapping dict
    for index,row in tqdm(df.iterrows(), total=len(df)):
        row_dict = row.to_dict()
        id2type[str(int(row_dict['_id']))] = row_dict['_labels'][1:]

    # filter the edges to 30 relation types
    if args.clean_rels:
        tri_rels = tri_df['_type'].value_counts()
        top30_types = tri_rels.nlargest(args.rel_num).index
        filtered_df = tri_df[tri_df['_type'].isin(top30_types)]
    else:
        filtered_df = tri_df
        
    # construct the graph based on the filterd dataframe.
    print(f"Constructing the knowledge graph as networkx object..")
    G = nx.DiGraph()
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        G.add_edge(row['_start'], row['_end'], relation=row['_type'])

    print(f"Graph construction finished, starting to get graph attributes. ")
    all_results = []
    graph_attributes = get_graph_attributes(G, filtered_df)
    graph_attributes['start_size'] = 0
    graph_attributes['sample_ratio'] = 0
    all_results.append(graph_attributes)

    # intialize the starting nodes  
    if args.target_types:  
        start_nodes = [k for k,v in id2type.items() if v in args.target_types and k in G]
    else:
        start_nodes = [k for k,v in id2type.items() if k in G]
    # keep the target relations
    if args.keep_target_rels:
        target_edges = []
        for index, row in tri_df.iterrows():
            # if (row['_type'] in args.target_rels) and (row['_start'] in G) and (row['_end'] in G):
            if row['_type'] in args.target_rels:
                target_edges.append(row)
    else:
        target_edges = []
    
    print(f"The size of start nodes is {len(start_nodes)}")
    # if start_size is a list, then only keep the stats rather than real sampling
    if isinstance(args.start_size, list):
        for start_size in args.start_size:
            sampled_start_nodes = random.sample(start_nodes, start_size)
    
            sample_size = int(len(filtered_df) * args.sample_ratio)
            print(f"Starting random walk, target edge size is {sample_size}")

            sampled_edges = random_walk_uni(G, sampled_start_nodes, target_edge_count=sample_size)

            sampled_results = target_edges if target_edges else []
            for _, row in filtered_df.iterrows():
                if (row['_start'], row['_end']) in sampled_edges:
                    sampled_results.append(row)

            sampled_df = pd.DataFrame(sampled_results)

            G_sub = nx.DiGraph()
            for _, row in sampled_df.iterrows():
                G_sub.add_edge(row['_start'], row['_end'], relation=row['_type'])

            # Get graph attributes and add experiment-specific info (start_size and sample_ratio)
            graph_attributes = get_graph_attributes(G_sub, sampled_df)
            graph_attributes['start_size'] = start_size
            graph_attributes['sample_ratio'] = args.sample_ratio

            # Append the result to all_results
            all_results.append(graph_attributes)

        if args.save_to_csv:
            output_df = pd.DataFrame(all_results)
            output_df.to_csv(args.out_stat_path, index=False)
            print(f"Results saved to {args.out_stat_path}")
    else:
        sampled_start_nodes = random.sample(start_nodes, args.start_size)
        sample_size = int(len(filtered_df) * args.sample_ratio)
        print(f"Starting random walk, target edge size is {sample_size}")

        sampled_edges = random_walk_uni(G, sampled_start_nodes, target_edge_count=sample_size)

        sampled_results = target_edges if target_edges else []
        for _, row in filtered_df.iterrows():
            if (row['_start'], row['_end']) in sampled_edges:
                sampled_results.append(row)

        sampled_df = pd.DataFrame(sampled_results)
        subgraph_edges = sampled_df[['_start', '_type', '_end']]
        print(f'Saving sampled edges to {args.out_path}')

        # Save to CSV (without index)
        subgraph_edges.to_csv(args.out_path, index=False, header=False, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_rels', default=False, help="Whether to filter out the most frequent rels.")
    parser.add_argument('--keep_target_rels', default=False)
    parser.add_argument('--rel_num', default=30)
    # parser.add_argument('--target_types', default=['Chemical:Compound', 'Disease'], help="The starting node types.")
    parser.add_argument('--target_types', default=None, help="The starting node types.")
    parser.add_argument('--target_rels', default=['TREATS_CHtD', 'INDUCES_CHiD'], help="The target relation that are all kept.")
    parser.add_argument('--start_size', default=[2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000, 30000, 40000, 50000])
    # parser.add_argument('--start_size', default=[1000,2000,4000,6000,8000])
    # 1000 for target identification, 2000 for drug repo
    # parser.add_argument('--start_size', default=2000)
    # parser.add_argument('--sample_ratio', default=0.005)
    parser.add_argument('--sample_ratio', default=0.01)

    parser.add_argument('--save_to_csv', default=True)
    parser.add_argument('--tri_path', default='/dataStor/home/yyma/KGC-project/CM-BKG/data/remove_assay/train.csv')
    parser.add_argument('--out_stat_path', default='/dataStor/home/yyma/KGC-project/CM-BKG/src/graph_partition/random_walk_stat.csv')
    parser.add_argument('--out_path', default='/dataStor/home/yyma/KGC-project/CM-BKG/data/partitions/inductive/train.csv')

    args = parser.parse_args()

    main(args)