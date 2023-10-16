import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import networkx as nx
import plotly.graph_objects as go

num_of_nodes = 1000

# Barabási-Albert
num_of_edges = 5
ba_graph = nx.barabasi_albert_graph(num_of_nodes, num_of_edges)
ba_model = ep.SIRModel(ba_graph)

# Watts-Strogatz
k_nearest_neighbors = 6
rewiring_probability = 0.1
ws_graph = nx.watts_strogatz_graph(num_of_nodes, k_nearest_neighbors, rewiring_probability)
ws_model = ep.SIRModel(ws_graph)

# Erdős-Rényi
edge_creation_probability = 0.1
er_graph = nx.erdos_renyi_graph(num_of_nodes, edge_creation_probability)
er_model = ep.SIRModel(er_graph)

# Parameters to vary
betas = [0.05, 0.1, 0.3]
gammas = [0.005, 0.1]

results = {}

for beta in betas:
    for gamma in gammas:
        config = mc.Configuration()
        config.add_model_parameter('beta', beta)
        config.add_model_parameter('gamma', gamma)
        config.add_model_initial_configuration("Infected", [i for i in range(int(0.1*1000))]) # 10% initially infected

        for graph, name in [(ba_graph, "Barabási-Albert"), (ws_graph, "Watts-Strogatz"), (er_graph, "Erdős-Rényi")]:
            model = ep.SIRModel(graph)
            model.set_initial_status(config)
            iterations = model.iteration_bunch(200)
            
            key = (name, beta, gamma)
            results[key] = iterations

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the web application
app.layout = html.Div([
    dcc.Graph(id='network-graph'),
    dcc.Slider(
        id='time-slider',
        min=0,
        max=len(iterations)-1,
        value=0,
        marks={i: str(i) for i in range(0, len(iterations), 10)},
        step=1
    )
])


@app.callback(
    Output('network-graph', 'figure'),
    [Input('time-slider', 'value')]
)
def update_graph(selected_time):


    pos = nx.spring_layout(ba_graph)
    edge_x = []
    edge_y = []
    for edge in ba_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in ba_graph.nodes()]
    node_y = [pos[node][1] for node in ba_graph.nodes()]

    #Colors
    default_beta = 0.05
    default_gamma = 0.005
    model_network = "Barabási-Albert"
    key = (model_network, default_beta, default_gamma)
    current_status = results[key][selected_time]['status']

    node_colors = []
    for node, status in current_status.items():
        if status == 0:  
            node_colors.append("yellow")
        elif status == 1:  
            node_colors.append("red")
        else:  
            node_colors.append("green")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines"
            ),
            go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=True,
                    colorscale="Viridis",
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title="Node Connections",
                        xanchor="left",
                        titleside="right"
                    ),
                    line_width=2,
                    color=node_colors
                )
            )
        ],
        layout=go.Layout(
            title="Network Graph with Disease Progression",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
