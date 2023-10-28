import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random
import pdb

data = pd.read_csv('../transmission_network.csv', delimiter=';', index_col=0) 

G = nx.Graph()
G.add_nodes_from(data.index.tolist())
G.add_nodes_from(data.columns.astype(int).tolist())

rows = data.index.tolist()
columns = data.columns.astype(int).tolist()

if G.number_of_nodes() != 374:
    raise ValueError("The number of nodes is not 364.")

for i, row in enumerate(rows):
    for j, col in enumerate(columns):
        value = data.iloc[i, j]
        if value > 0:
            G.add_edge(row, col, weight=value)
            
if  G.number_of_edges() != 1265:
    raise ValueError("The number of edges is not 1265.")

def dynamic_vaccination_strategy(graph, model, budget, vaccination_budget, test_accuracy):
    node_degrees = dict(graph.degree())
    sorted_nodes_by_high_degree = sorted(node_degrees, key=node_degrees.get, reverse=True)

    tests_run = 0

    while tests_run < budget:
        for node in sorted_nodes_by_high_degree:
            if tests_run < budget:
                if random.random() < test_accuracy:
                    node_status = model.status[node]
                    # If the node is susceptible, vaccinate
                    if node_status == 0:
                        model.status[node] = 2  # Move to removed state
                        tests_run += 1
            else:
                break

def run_simulation(graph, total_tests, vaccination_budget, test_accuracy):
    model = ep.SIRModel(graph)
    config = mc.Configuration()
    config.add_model_parameter('beta', 0.3)
    config.add_model_parameter('gamma', 0.1)
    config.add_model_initial_configuration("Infected", random.sample(list(graph.nodes()), 5))
    model.set_initial_status(config)
    
    dynamic_vaccination_strategy(graph, model, total_tests, vaccination_budget, test_accuracy)
    iterations = model.iteration_bunch(200)
    infected_dynamic = sum([it['node_count'][1] for it in iterations]) / 200
    
    # Random strategy
    model.reset()
    model.set_initial_status(config)
    for _ in range(vaccination_budget):
        random_node = random.choice(list(graph.nodes()))
        model.status[random_node] = 2
    iterations = model.iteration_bunch(200)
    infected_random = sum([it['node_count'][1] for it in iterations]) / 200

    return infected_dynamic, infected_random

budgets = [1, 3, 5, 10]
accuracies = [0.5, 0.75, 1.0]

for budget in budgets:
    for accuracy in accuracies:
        dynamic_infected, random_infected = run_simulation(G, 200, budget, accuracy)
        print(f"Budget: {budget}, Accuracy: {accuracy}")
        print(f"Dynamic Vaccination Strategy Average Infected: {dynamic_infected}")
        print(f"Random Strategy Average Infected: {random_infected}")

