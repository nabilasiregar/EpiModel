import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import matplotlib.pyplot as plt
import numpy as np

num_of_nodes = 1000
network1 = "Watts-Strogatz"
network2 = "Barabási-Albert"
network3 = "Erdős-Rényi"

# Network parameters
k_nearest_neighbors = 6
rewiring_probability = 0.1
num_of_edges = 5
edge_creation_probability = 0.1

# Disease parameters
beta = 0.3
gamma = 0.1

ws_graph = nx.watts_strogatz_graph(num_of_nodes, k_nearest_neighbors, rewiring_probability)
ba_graph = nx.barabasi_albert_graph(num_of_nodes, num_of_edges)
er_graph = nx.erdos_renyi_graph(num_of_nodes, edge_creation_probability)

config = mc.Configuration()
config.add_model_parameter('beta', beta)
config.add_model_parameter('gamma', gamma)
config.add_model_parameter("percentage_infected", 0.1) # 10% initially infected

def get_graph(network_type):
    if network_type == 'Watts-Strogatz':
        graph = ws_graph
    elif network_type == 'Barabási-Albert':
        graph = ba_graph
    elif network_type == 'Erdős-Rényi':
        graph = er_graph
    
    return graph

def get_avg_shortest_path_length(model):
    return nx.average_shortest_path_length(model)


ws_stats = get_avg_shortest_path_length(get_graph(network1))
ba_stats = get_avg_shortest_path_length(get_graph(network2))
er_stats = get_avg_shortest_path_length(get_graph(network3))