import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, gridplot
from bokeh.models import HoverTool

def create_network(network_type, *args):
    if network_type == "BarabasiAlbert":
        m = args[0]
        return nx.barabasi_albert_graph(1000, m)
    elif network_type == "WattsStrogatz":
        k, p = args
        return nx.watts_strogatz_graph(1000, k, p)
    elif network_type == "ErdosReyni":
        p = args[0]
        return nx.erdos_renyi_graph(1000, p)
    else:
        raise ValueError("Invalid network type")

def simulate_SIR(network, beta, gamma):

    model = ep.SIRModel(network)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_parameter("fraction_infected", 0.05)
    model.set_initial_status(config)

    # Simulation
    iterations = model.iteration_bunch(200)
    return iterations

def plot_results_matplotlib(ax, results, network_type, beta, gamma):
    susceptible = [iteration['node_count'][0] for iteration in results]
    infected = [iteration['node_count'][1] for iteration in results]
    recovered = [iteration['node_count'][2] for iteration in results]

    ax.plot(susceptible, label="Susceptible", color="blue")
    ax.plot(infected, label="Infected", color="red")
    ax.plot(recovered, label="Recovered", color="green")
    ax.set_title(f"{network_type}\nBeta: {beta}, Gamma: {gamma}")

def plot_results_bokeh(results, network_type, beta, gamma):
    susceptible = [iteration['node_count'][0] for iteration in results]
    infected = [iteration['node_count'][1] for iteration in results]
    recovered = [iteration['node_count'][2] for iteration in results]

    p = figure(width=300, height=300, title=f"{network_type}\nBeta: {beta}, Gamma: {gamma}")
    p.line(list(range(len(susceptible))), susceptible, legend_label="Susceptible", color="blue")
    p.line(list(range(len(infected))), infected, legend_label="Infected", color="red")
    p.line(list(range(len(recovered))), recovered, legend_label="Recovered", color="green")
    p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[('Value', '$y')]))
    return p

def main(visualization="matplotlib"):
    network_types = ["BarabasiAlbert", "WattsStrogatz", "ErdosReyni"]
    betas = [0.01, 0.05, 0.1]
    gammas = [0.01, 0.05, 0.1]

    if visualization == "matplotlib":
        fig, axes = plt.subplots(len(network_types), len(betas) * len(gammas), figsize=(15, 9))
        plt.tight_layout(pad=3.0)

        for i, network_type in enumerate(network_types):
                if network_type == "BarabasiAlbert":
                    network = create_network(network_type, 5)
                elif network_type == "WattsStrogatz":
                    network = create_network(network_type, 6, 0.05)
                else:
                    network = create_network(network_type, 0.05)


                for j, beta in enumerate(betas):
                    for k, gamma in enumerate(gammas):
                        results = simulate_SIR(network, beta, gamma)
                        ax = axes[i][j * len(gammas) + k]
                        plot_results_matplotlib(ax, results, network_type, beta, gamma)

        fig.text(0.5, 0.04, 'Iterations', ha='center')
        fig.text(0.04, 0.5, 'Node Count', va='center', rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.show()

    elif visualization == "bokeh":
        plots = []

        for network_type in network_types:
            row = []
            if network_type == "BarabasiAlbert":
                network = create_network(network_type, 5)
            elif network_type == "WattsStrogatz":
                network = create_network(network_type, 6, 0.05)
            else:
                network = create_network(network_type, 0.05)

            for beta in betas:
                for gamma in gammas:
                    results = simulate_SIR(network, beta, gamma)
                    p = plot_results_bokeh(results, network_type, beta, gamma)
                    row.append(p)
            plots.append(row)

        grid = gridplot(plots)
        show(grid)


if __name__ == "__main__":
    main(visualization="matplotlib")