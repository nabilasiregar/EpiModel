import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
from bokeh.plotting import figure, show, curdoc
from bokeh.models import Slider, ColumnDataSource, HoverTool
from bokeh.layouts import column, row

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

def plot_results_bokeh(source, title):
    p = figure(width=300, height=300, title=title)
    p.line(x='x', y='susceptible', source=source, legend_label="Susceptible", color="blue")
    p.line(x='x', y='infected', source=source, legend_label="Infected", color="red")
    p.line(x='x', y='recovered', source=source, legend_label="Recovered", color="green")
    p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[('Value', '$y')]))
    return p

def update(attr, old, new):
    beta = beta_slider.value
    gamma = gamma_slider.value
    for network_type, source in sources.items():
        if network_type == "BarabasiAlbert":
            network = create_network(network_type, 5)
        elif network_type == "WattsStrogatz":
            network = create_network(network_type, 6, 0.05)
        else:
            network = create_network(network_type, 0.05)
        results = simulate_SIR(network, beta, gamma)
        source.data = {
            'x': list(range(len(results))),
            'susceptible': [iteration['node_count'][0] for iteration in results],
            'infected': [iteration['node_count'][1] for iteration in results],
            'recovered': [iteration['node_count'][2] for iteration in results]
        }

# Create Sliders
beta_slider = Slider(start=0.01, end=0.1, value=0.05, step=0.01, title="Beta")
gamma_slider = Slider(start=0.01, end=0.1, value=0.05, step=0.01, title="Gamma")

beta_slider.on_change('value', update)
gamma_slider.on_change('value', update)

network_types = ["BarabasiAlbert", "WattsStrogatz", "ErdosReyni"]
plots = []
sources = {}

for network_type in network_types:
    source = ColumnDataSource(data={'x': [], 'susceptible': [], 'infected': [], 'recovered': []})
    sources[network_type] = source
    p = plot_results_bokeh(source, network_type)
    plots.append(p)

layout = column(beta_slider, gamma_slider, *plots)
curdoc().add_root(layout)
curdoc().title = "SIR Model Simulation"