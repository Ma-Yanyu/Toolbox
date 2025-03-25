# based on the setting
# either output experiments stats or sampled edges.

import pandas as pd
import argparse
from tqdm import tqdm
import random
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict

# perform random walk with uniform distribution applied on relations
def random_walk_uni(G, start_nodes, restart_prob=0.15, target_edge_count=1_000):
    
    start_nodes_in_graph = [node for node in start_nodes if node in G]
    
    # Make sure there are valid start nodes
    if not start_nodes_in_graph:
        raise ValueError("No valid start nodes found in the graph.")
    
    current_node = random.choice(start_nodes_in_graph) 
    sampled_edges = set()  
    sampled_nodes = set([current_node])  # To store the sampled nodes

    edge_relation_map = nx.get_edge_attributes(G, 'relation')

    with tqdm(total=target_edge_count, desc="Sampling edges", unit="edge") as pbar:
        while len(sampled_edges) < target_edge_count:
            # With restart probability, jump back to one of the start nodes
            if random.random() < restart_prob:
                current_node = random.choice(start_nodes_in_graph)  # Ensure we choose from valid nodes
            else:
                # Get all outgoing neighbors and corresponding edges
                # neighbors = list(G.neighbors(current_node))
                # Outgoing edges: (current_node, neighbor)
                neighbors_out = list(G.successors(current_node))
                # Incoming edges: (neighbor, current_node)
                neighbors_in = list(G.predecessors(current_node))

                neighbors = []
                for v in neighbors_out:
                    neighbors.append((current_node, v))
                for u in neighbors_in:
                    neighbors.append((u, current_node))

                if neighbors:
                    # Group neighbors by relation type
                    edges_by_relation = {}
                    for u, v in neighbors:
                        relation = edge_relation_map.get((u, v))
                        if relation not in edges_by_relation:
                            edges_by_relation[relation] = []
                        edges_by_relation[relation].append((u, v))
                    
                    # Uniformly choose a relation type from the available ones
                    available_relations = list(edges_by_relation.keys())
                    chosen_relation = random.choice(available_relations)
                    
                    # Uniformly choose an edge that corresponds to the chosen relation type
                    u, v = random.choice(edges_by_relation[chosen_relation])
                    
                    # Check if the edge is already sampled, if not, add it to sampled edges
                    if (u, v) not in sampled_edges:
                        sampled_edges.add((u, v))  
                        pbar.update(1)  
                    current_node = v 
    
    return sampled_edges

def get_n_hop_edges(G, target_edges, hop):
    """Expand n-hop neighbors around target edges without relation filtering."""
    # Collect source and target nodes from target edges
    target_nodes = set()
    for u, v, _ in target_edges:
        target_nodes.update([u, v])

    # Expand neighbors hop by hop
    expanded_nodes = set(target_nodes)
    expanded_edges = set(target_edges)

    for _ in range(hop):
        new_neighbors = set()
        for node in expanded_nodes:
            # Collect predecessors and successors without relation filtering
            for neighbor in G.predecessors(node):
                new_neighbors.add(neighbor)
                expanded_edges.add((neighbor, node, G[neighbor][node]['relation']))
            
            for neighbor in G.successors(node):
                new_neighbors.add(neighbor)
                expanded_edges.add((node, neighbor, G[node][neighbor]['relation']))

        # Update the expanded nodes for the next hop
        expanded_nodes.update(new_neighbors)

    return expanded_edges

# record graph stats
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

    # Graph density
    density = nx.density(G_sub)
    results['density'] = density

    # Number of nodes and edges
    results['num_nodes'] = G_sub.number_of_nodes()
    results['num_edges'] = G_sub.number_of_edges()

    return results

def main(args):
    df = pd.read_csv(Path(args.root_path) / 'data/raw/extracted_data_entity_text_yz.csv')
    
    tri_df = pd.read_csv(Path(args.root_path) / args.tri_path)
    tri_df['_start'] = tri_df['_start'].apply(lambda x: str(int(x)))
    tri_df['_end'] = tri_df['_end'].apply(lambda x: str(int(x)))

    id2type = defaultdict()
    # create a node-to-type mapping dict
    for _,row in tqdm(df.iterrows(), total=len(df)):
        row_dict = row.to_dict()
        id2type[str(int(row_dict['_id']))] = row_dict['_labels'][1:]
        
    # construct the graph based on the filterd dataframe.
    print(f"Constructing the knowledge graph as networkx object..")
    G = nx.DiGraph()
    for _, row in tqdm(tri_df.iterrows(), total=len(tri_df)):
        G.add_edge(row['_start'], row['_end'], relation=row['_type'])

    print(f"Graph construction finished, starting to get graph attributes. ")
    all_results = []
    graph_attributes = get_graph_attributes(G, tri_df)
    graph_attributes['start_size'] = 0
    graph_attributes['sample_ratio'] = 0
    all_results.append(graph_attributes)

    if isinstance(args.target_relation, list):
        existing_relations = {data['relation'] for _, _, data in G.edges(data=True)}
        missing_relations = [rel for rel in args.target_relation if rel not in existing_relations]

        if missing_relations:
            raise ValueError(f"❌ The following target relations were not found in the graph: {missing_relations}")
        target_edges = [(u, v, d['relation']) for u, v, d in G.edges(data=True)
                    if d['relation'] in args.target_relation]
    else:
        # Check if single target relation exists
        if args.target_relation not in {data['relation'] for _, _, data in G.edges(data=True)}:
            raise ValueError(f"❌ The target relation '{args.target_relation}' was not found in the graph.")

        # Collect target edges for a single relation
        target_edges = [(u, v, d['relation']) for u, v, d in G.edges(data=True)
                        if d['relation'] == args.target_relation]
    
    # collect its neighbor info and threshold the relation types
    result_edges = get_n_hop_edges(G, target_edges, 1)
    subgraph_edges = pd.DataFrame(result_edges, columns=['_start', '_end', '_type'])
    tri_rels = subgraph_edges['_type'].value_counts()
    top30_types = tri_rels.nlargest(args.rel_num).index
    filtered_df = tri_df[tri_df['_type'].isin(top30_types)]

    # remove edges from the graph
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] not in set(top30_types)]
    G.remove_edges_from(edges_to_remove)

    """
    variables table:
    start_node_pairs - set of pairs of nodes belonging to the target edges;
    start_nodes - list of nodes linked to the targeted edges;
    sampled_start_edges - edges that are initially sampled as

    """
    # intialize the starting nodes  
    start_node_pairs = set([(u, v) for u, v, _ in target_edges])

    # Flatten the set to ensure unique nodes are stored
    start_nodes = set(u for u, v in start_node_pairs) | set(v for u, v in start_node_pairs)
    start_nodes = sorted(start_nodes)

    if isinstance(args.node_sample_ratio, list):
        for start_size in args.node_sample_ratio:
            sampled_results = []
            sampled_target_edges = []
            sampled_start_nodes = set()
            start_size = int(len(start_nodes) * float(start_size))

            current_node_pairs = start_node_pairs.copy()
            while len(sampled_start_nodes) < start_size and start_node_pairs:
                u, v = random.choice(list(current_node_pairs))
                current_node_pairs.remove((u, v))
                sampled_start_nodes.update([u, v])
                sampled_target_edges.append((u,v))
    
            sample_size = int(len(filtered_df) * args.edge_sample_ratio)
            print(f"Starting random walk, target edge size is {sample_size}")

            sampled_edges = list(random_walk_uni(G, sampled_start_nodes, target_edge_count=sample_size))
            sampled_edges.extend(sampled_target_edges)
            print(f"Total {len(sampled_edges)} edges sampled, including {len(sampled_target_edges)} target edges.")

            # sampled_df = filtered_df[filtered_df.apply(lambda row: (row['_start'], row['_end']) in sampled_edges, axis=1)]
            sampled_edges = set(sampled_edges)
            for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
                if (row['_start'], row['_end']) in sampled_edges:
                    sampled_results.append(row)

            sampled_df = pd.DataFrame(sampled_results, columns=['_start', '_end', '_type'])

            print("Constructing graph object...")
            G_sub = nx.DiGraph()
            for row in tqdm(sampled_df.itertuples(index=False)):
                G_sub.add_edge(row[0], row[1], relation=row[2])

            # Get graph attributes and add experiment-specific info (start_size and sample_ratio)
            print("Caculating graph properties...")
            graph_attributes = get_graph_attributes(G_sub, sampled_df)
            graph_attributes['start_size'] = start_size
            graph_attributes['sample_ratio'] = args.node_sample_ratio

            # Append the result to all_results
            all_results.append(graph_attributes)

        output_path = Path(args.root_path) / args.out_stat_path
        if args.save_to_csv:
            output_df = pd.DataFrame(all_results)
            output_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
    else:
        sampled_results = []
        sampled_target_edges = []
        sampled_start_nodes = set()
        start_size = int(len(start_nodes) * float(args.node_sample_ratio))

        while len(sampled_start_nodes) < start_size and start_node_pairs:
            u, v = random.choice(list(start_node_pairs))
            start_node_pairs.remove((u, v))
            sampled_start_nodes.update([u, v])
            sampled_target_edges.append((u,v))

        sample_size = int(len(filtered_df) * args.edge_sample_ratio)
        print(f"Starting random walk, target edge size is {sample_size}")

        sampled_edges = list(random_walk_uni(G, sampled_start_nodes, target_edge_count=sample_size))
        sampled_edges.extend(sampled_target_edges)
        print(f"Total {len(sampled_edges)} edges sampled, including {len(sampled_target_edges)} target edges.")

        # sampled_df = filtered_df[filtered_df.apply(lambda row: (row['_start'], row['_end']) in sampled_edges, axis=1)]
        sampled_edges = set(sampled_edges)
        for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
            if (row['_start'], row['_end']) in sampled_edges:
                sampled_results.append(row)

        sampled_df = pd.DataFrame(sampled_results, columns=['_start', '_end', '_type'])
        subgraph_edges = sampled_df[['_start', '_type', '_end']]
        
        # Save to CSV (without index)
        output_path = Path(args.root_path) / args.out_path
        print(f'Saving sampled edges to {args.out_path}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subgraph_edges.to_csv(output_path, index=False, header=False, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_rels', default=False, help="Whether to filter out the infrequent rels.")
    parser.add_argument('--rel_num', default=30)
    parser.add_argument('--target_relation', default=['TREATS_CHtD'], 
                        help="If not none, target relation is all kept.")
                    
    parser.add_argument('--node_sample_ratio', default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 , 0.9, 1.0], 
                        help="If its an integer, random-walk parition is performed; If its a list of integers, then we only gather stats.")
    # parser.add_argument('--node_sample_ratio', default=0.6, 
    #                     help="If its an integer, random-walk parition is performed; If its a list of integers, then we only gather stats.")
    parser.add_argument('--edge_sample_ratio', default=0.005, help="The edge sample ratio")

    parser.add_argument('--save_to_csv', default=True)
    parser.add_argument('--root_path', default='/dataStor/home/yyma/KGC-project/CM-BKG')
    parser.add_argument('--tri_path', default='data/remove_assay/train.csv')
    parser.add_argument('--out_stat_path', default='src/graph_partition/random_walk_stat.csv')
    parser.add_argument('--out_path', default='data/cross_val/task2/0/train.csv')
    args = parser.parse_args()

    main(args)