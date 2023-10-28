import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pdb

data = open('../transmission_network.csv', 'rb')
graph = nx.read_multiline_adjlist(data, delimiter=';', nodetype=int)
pdb.set_trace()

# for i, row in enumerate(adj_matrix):
#     source = node_labels[i]
#     for j, weight in enumerate(row):
#         target = node_labels[j]
#         if weight > 0:  # only edges with positive weights
#             G.add_edge(source, target, weight=weight)

# # Model selection
# model = ep.SIRModel(G)

# # Model Configuration
# config = mc.Configuration()
# config.add_model_parameter('beta', 0.01)
# config.add_model_parameter('gamma', 0.005)
# config.add_model_parameter("fraction_infected", 0.2) 

# model.set_initial_status(config)

# # Simulation execution
# iterations = model.iteration_bunch(200)

# class Graph:
#     def __init__(self, num_of_nodes):
#         self.num_of_nodes = num_of_nodes

#     def barabasi_albert(self, num_of_edges):
#         return nx.barabasi_albert_graph(self.num_of_nodes, num_of_edges)

#     def watts_strogatz(self, k_nearest_neighbors, rewiring_probability):
#         return nx.watts_strogatz_graph(self.num_of_nodes, k_nearest_neighbors, rewiring_probability)

#     def erdos_renyi(self, edge_creation_probability):
#         return nx.erdos_renyi_graph(self.num_of_nodes, edge_creation_probability)

# def get_cumulative_status(time_step):
#     all_status = {}
#     for t in range(time_step + 1):
#         all_status.update(iterations[t]['status'])
#     return all_status

# def show_graph(time_step):
#     current_status = get_cumulative_status(time_step)
#     colors = [ "green" if current_status.get(node) == 2 
#                else "red" if current_status.get(node) == 1 
#                else "yellow" for node in G.nodes()]

#     nx.draw(model, node_color=colors, with_labels=False)
#     plt.show()

# # Usage
# time_step = 3

# show_graph(time_step)


