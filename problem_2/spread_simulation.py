import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from ndlib.models.ContinuousModel import ContinuousModel
from ndlib.models.compartments.NodeStochastic import NodeStochastic
import ndlib.models.ModelConfig as mc

################### MODEL SPECIFICATIONS ###################

constants = {
    'beta': 0.1,  
    'gamma': 0.01
}

# Initial status
initial_status = {
    'S': 0.9,
    'I': 0.1,
    'R': 0
}

def update_S(node, graph, status, attributes, constants):
    infected_neighbors = sum([1 for n in graph.neighbors(node) if status[n]['I'] == 1])
    return status[node]['S'] - constants['beta'] * status[node]['S'] * infected_neighbors

def update_I(node, graph, status, attributes, constants):
    return status[node]['I'] + constants['beta'] * status[node]['S'] - constants['gamma'] * status[node]['I']

def update_R(node, graph, status, attributes, constants):
    return status[node]['R'] + constants['gamma'] * status[node]['I']

################### MODEL CONFIGURATION ###################

ba_graph = nx.barabasi_albert_graph(100, 5)

visualization_config = {
    'plot_interval': 2,
    'plot_variable': 'I',
    'variable_limits': {
        'S': [0, 1],
        'I': [0, 1],
        'R': [0, 1]
    },
    'show_plot': True,
    'plot_output': '../assets/disease_spread.gif',
    'plot_title': 'Disease Spread on a BA Network',
}

model = ContinuousModel(ba_graph, constants=constants)
model.add_status('S')
model.add_status('I')
model.add_status('R')

condition = NodeStochastic(1)
model.add_rule('S', update_S, condition)
model.add_rule('I', update_I, condition)
model.add_rule('R', update_R, condition)

config = mc.Configuration()
model.set_initial_status(initial_status, config)
model.configure_visualization(visualization_config)

################### SIMULATION ###################

iterations = model.iteration_bunch(100, node_status=True)
trends = model.build_trends(iterations)

################### VISUALIZATION ###################

model.plot(trends, len(iterations), delta=True)

x = np.arange(0, len(iterations))
plt.figure()

plt.subplot(221)
plt.plot(x, trends['means']['S'], label='S')
plt.legend()

plt.subplot(222)
plt.plot(x, trends['means']['I'], label='I')
plt.legend()

plt.subplot(223)
plt.plot(x, trends['means']['R'], label='R')
plt.legend()

plt.show()

model.visualize(iterations)
