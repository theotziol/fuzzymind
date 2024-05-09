import streamlit as st 
from matplotlib import pyplot as plt
import numpy as np

import io 
import networkx as nx

#help_label_pos = 'This is a parameter used to define the location of the label, with 0.5 at center (avoid to to ovelap)'
help_DPI = 'The DPI value of the graph'

@st.cache_data
def create_graph(edited_matrix, figsize, nodesize, font_size, weights_font_size, title_font_size, arrowsize,):
    fig = st.pyplot(create_visual_map(edited_matrix, figsize, nodesize, font_size, weights_font_size, title_font_size, arrowsize), clear_figure=False)

def graph(edited_matrix):
    st.subheader('FCM graph')
    col1, col2 = st.columns(2, gap = 'small')
    with col1:
        #this column is for modifying the graph parameters
        figsize = st.slider('Figure size', 5, 20, 10, 1)
        nodesize = st.slider('Node size', 300, 4000, 1500, 100)

        # check if needed
        # label_pos = st.slider('Node size', 0.1, 0.9, 0.4, 0.1) 
        # font_size = st.slider('Font size (nodes)', 3, 40, 10, 1)
        # weights_font_size = st.slider('Font size (weights)', 5, 30, 10, 1)
        # title_font_size = st.slider('Font size (title)', 10, 100, 30, 5)

        dpi = st.slider('DPI', 300, 1500, 600, 100, help=help_DPI)
        font_size = nodesize // 200
        weights_font_size = font_size
        arrowsize = weights_font_size + 1 
        title_font_size = figsize*2     
        
    with col2:
        #this column is for the graph
        fig = create_graph(edited_matrix, figsize, nodesize, font_size, weights_font_size, title_font_size, arrowsize)
    
    fn = 'FCM_graph.png'
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi = dpi)

    btn = st.download_button('Download figure', data = img, file_name=fn, mime="image/png")
    if btn:
        plt.close('all')


def create_visual_map(
    df,
    figsize = 10,
    node_size = 1000,
    font_size = 6,
    weight_font_size = 6,
    title_font_size = 30,
    arrowsize = 10,
    ):
    '''
    This function creates the fcm graph based on the networkx library
    '''
    #todo add more layouts with a widget
    #todo add more colomaps and colors with a widget
    df = df.transpose() #df is transposed due to column-wise arrows of the graph
    
    G = nx.MultiDiGraph()
    for i in df.columns:
        for j in df.columns:
            weight = df[i].loc[j]
            if weight != 0.0:
                G.add_edge(i, j, weight = weight)


    fig, ax = plt.subplots(figsize=(figsize + figsize//3, figsize))
    pos = nx.circular_layout(G)  # Define layout. Alternative: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
    M = G.number_of_edges()
    colors = [i[2]['weight'] for i in G.edges(data=True)]

    options = {
    "node_color": 'skyblue',
    "edge_color": list(np.abs(colors)),
    "width": 4,
    "edge_cmap": plt.cm.Blues,
    "with_labels": True,
    "node_size" : node_size,
    "font_size" : font_size,
    "arrows" : True,
    "arrowstyle" : 'fancy',
    "arrowsize" : arrowsize,
    "connectionstyle" : 'arc3, rad = 0.1',
    "edge_vmin" : 0,
    "edge_vmax": 1,
    }
    # nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='skyblue', font_size=font_size)  # Adjust node_size and font_size
    
    nx.draw(G, pos, **options)
    # Add edge labels (weights)
    edge_labels = {(n1, n2): d['weight'] for n1, n2, d in G.edges(data=True)}
    labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=weight_font_size, label_pos=0.4, connectionstyle='arc3, rad = 0.1')  # Adjust font_size for edge labels
    sm = plt.cm.ScalarMappable(cmap = plt.cm.Blues, norm = plt.Normalize(0, 1))
    sm._A = []
    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    cbar = plt.colorbar(sm, ticks = ticks, shrink = 0.95, fraction = 0.1, ax = ax) 
    cbar.ax.set_yticklabels([f'$\\pm {i}$' if i != 0 else '0' for i in ticks])
    cbar.ax.tick_params(labelsize = font_size)
    plt.title('Fuzzy Cognitive Map', fontsize = title_font_size + 1)
    plt.tight_layout()  # Ensure tight layout
    return fig