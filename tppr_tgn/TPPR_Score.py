import random
from collections import defaultdict
import time
import numpy as np
import csv
import tppr_tgn.csv_to_text as csv_to_text

def extract_edges_from_txt(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, timestamp = parts
                edges.append((int(node1), int(node2), int(timestamp)))

     # مرتب‌سازی edges بر اساس timestamp (قسمت سوم هر تپل)
    sorted_edges = sorted(edges, key=lambda x: x[2])

    return sorted_edges

# Parse the edge list into a dictionary
def parse_edge_list(edge_list):
    graph = defaultdict(list)
    for from_node, to_node, time in edge_list:
        graph[from_node].append((to_node, time))
    return graph

# Random Walk with Restart for a single start node
def random_walk_with_restart(graph, start_node, restart_prob, max_steps):
    current_node = start_node
    current_time = -1  # Start with the smallest possible time
    walk = [current_node]
    visit_counts = defaultdict(int)  # To count visits to each node
    visit_counts[current_node] += 1  # Count the starting node

    for _ in range(max_steps):
        if random.random() < restart_prob:
            # Restart the walk from the start node
            current_node = start_node
            current_time = 0
            #walk.append(current_node)
            #visit_counts[current_node] += 1
        else:
            # Get all possible next nodes with greater time
            possible_next = [(to_node, time, time-current_time) for to_node, time in graph[current_node] if time > current_time]
            sum_of_times = 0
            for to_node, time, dis_time in  possible_next :
                sum_of_times+=(1/dis_time)

            if not possible_next:
                # No valid next node, restart from the start node
                current_node = start_node
                current_time = 0
                #walk.append(current_node)
                #visit_counts[current_node] += 1
            else:
                # Randomly choose the next node
                next_node, next_time, next_dis_time = random.choice(possible_next)
                current_node = next_node
                current_time = next_time
                walk.append(current_node)
                visit_counts[current_node] += ((1/next_dis_time)/(sum_of_times))

    return walk, visit_counts

# Perform random walks for a list of start nodes
def random_walks_for_start_nodes(graph, start_nodes, restart_prob, max_steps):
    all_visit_counts = defaultdict(int)  # Aggregate visit counts across all walks

    for start_node in start_nodes:
        _, visit_counts = random_walk_with_restart(graph, start_node, restart_prob, max_steps)
        for node, count in visit_counts.items():
            all_visit_counts[node] += count  # Aggregate counts

    return all_visit_counts

# Normalize visit counts
def normalize_visit_counts(visit_counts):
    total_visits = sum(visit_counts.values())  # Total number of visits
    normalized_counts = {node: count / total_visits for node, count in visit_counts.items()}
    return normalized_counts

# Rank nodes by visit counts
def rank_nodes_by_visits(visit_counts):
    # Sort nodes by visit counts in descending order
    ranked_nodes = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked_nodes)

def tppr_main(dataset_name):
    # Example usage
    # dataset_name = 'wikipedia'
    csv_to_text(dataset_name, dataset_name + 'converted.txt')

    file_path = dataset_name + 'converted.txt'

    edge_list = extract_edges_from_txt(file_path)

    graph = parse_edge_list(edge_list)
    start_nodes = list(graph.keys())

    restart_prob = 0.4
    max_step = 1000

    # Perform random walks for all start nodes
    top_3_tppr_results = {}
    for start_node in start_nodes:
        index = start_node
        start_node = [start_node]
        all_visit_counts = random_walks_for_start_nodes(graph, start_node, restart_prob, max_step)

        # Normalize visit counts
        normalized_counts = normalize_visit_counts(all_visit_counts)

        # Rank nodes by visit counts
        ranked_nodes = rank_nodes_by_visits(normalized_counts)

        rank = sorted(normalized_counts, key=normalized_counts.get, reverse=True)
        # print("Ranked Nodes:", rank)

        # Get top 3 TPPR scores (node, score)
        top_3_tppr = dict(sorted(normalized_counts.items(), key=lambda x: x[1], reverse=True)[:3])

        top_3_tppr_results[index] = top_3_tppr

    # Use this top 3 tppr score to calculate the formule below:
    # biggest_score * embedding of respective node that come from another module + middle_score * embedding of respective node that come from another module + lowest_score * embedding of respective node that come from another module  
    return top_3_tppr_results

def csv_to_text(input_file, output_file):
    with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        next(reader, None)  # skip header
        for row in reader:
            first = row[0]
            second = row[1]
            third = str(int(float(row[2])))  # convert to int, handle "36.0" or "0.0"
            outfile.write(f"{first} {second} {third}\n")