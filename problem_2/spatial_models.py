import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import matplotlib.pyplot as plt
import pdb

# Network parameters
num_of_nodes = 30
num_of_edges = 5
k_nearest_neighbors = 6
rewiring_probability = 0.1
edge_creation_probability = 0.1

# Network topology
ba_graph = nx.barabasi_albert_graph(num_of_nodes, num_of_edges)
ws_graph = nx.watts_strogatz_graph(num_of_nodes, k_nearest_neighbors, rewiring_probability)
er_graph = nx.erdos_renyi_graph(num_of_nodes, edge_creation_probability)

# Disease Parameters
betas = [0.05, 0.1, 0.3]
gammas = [0.05, 0.1, 0.5]

results = {}

for beta in betas:
    for gamma in gammas:
        config = mc.Configuration()
        config.add_model_parameter('beta', beta)
        config.add_model_parameter('gamma', gamma)
        config.add_model_parameter('fraction_infected', 0.1) # 10% initially infected

        for graph, name in [(ba_graph, "Barabási-Albert"), (ws_graph, "Watts-Strogatz"), (er_graph, "Erdős-Rényi")]:
            model = ep.SIRModel(graph)
            model.set_initial_status(config)
            iterations = model.iteration_bunch(200)
            
            # Store results
            key = (name, beta, gamma)
            results[key] = iterations

def get_cumulative_status(key, time_step):
    all_status = {}
    for t in range(time_step + 1):
        all_status.update(results[key][t]['status'])
    return all_status

def show_graph(network_name, beta, gamma, time_step):
    if network_name == "Barabási-Albert":
        network = ba_graph
    elif network_name == "Watts-Strogatz":
        network = ws_graph
    elif network_name == "Erdős-Rényi":
        network = er_graph

    key = (network_name, beta, gamma)
    if key not in results:
        print(f"No results for {key}")
        return

    current_status = get_cumulative_status(key, time_step)
    colors = [ "green" if current_status.get(node) == 2 
               else "red" if current_status.get(node) == 1 
               else "yellow" for node in network.nodes()]

    nx.draw(network, node_color=colors, with_labels=False)
    filename = f"../assets/{network_name}_Beta{beta}_Gamma{gamma}_Time{time_step}.png"
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
    print(f"Graph saved as {filename}")

# Usage
beta = 0.3
gamma = 0.1
time_step = 3

show_graph("Barabási-Albert", beta, gamma, time_step)
show_graph("Erdős-Rényi", beta, gamma, time_step)
show_graph("Watts-Strogatz", beta, gamma, time_step)