# This script contains functions for visualizing FCM graphs based on networkx library documentation
# It is imported in FCM_Designer pages, as well as in Learning page to visualize the learned graphs

import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

import io
import networkx as nx


layouts = {
    "Circular": nx.circular_layout,
    "Random": nx.random_layout,
    "Shell": nx.shell_layout,
    "Spring": nx.spring_layout,
}

def graph(edited_matrix, linguistic=False):
    '''Main function that is called in software modules. 
    Using a two-column structure, it's purpose is 
    a: to render the widgets and to call either
    b: to call either the create_graph or the create_linguistic_graph'''
    st.subheader("FCM graph")
    col1, col2 = st.columns(2, gap="small")

    if linguistic:
        #spring layout doesn't work with linguistic weights
        list_layouts = list(layouts.keys())[:-1] 
    else:
        list_layouts = list(layouts.keys())
    with col1:
        # this column is for modifying the graph parameters
        layout = st.radio(
            "Change layout",
            list_layouts,
            key=f"position_{linguistic}",
            horizontal=True,
        )
        figsize = st.slider("Figure size", 5, 15, 10, 1, key=f"figsize_{linguistic}")
        nodesize = st.slider(
            "Node size", 600, 7000, 2400, 200, key=f"nodesize_{linguistic}"
        )

        layout_func = layouts[layout]
        dpi = 500 # dpi = st.slider('DPI', 300, 1500, 600, 100)
        font_size = nodesize // 200
        weights_font_size = font_size
        arrowsize = weights_font_size + 2
        title_font_size = figsize * 2

    with col2:
        # this column is for the graph
        if linguistic:
            fig = create_linguistic_graph(
                edited_matrix,
                layout_func,
                figsize,
                nodesize,
                font_size,
                weights_font_size,
                title_font_size,
                arrowsize,
            )
        else:
            fig = create_graph(
                edited_matrix,
                layout_func,
                figsize,
                nodesize,
                font_size,
                weights_font_size,
                title_font_size,
                arrowsize,
            )

    fn = "FCM_graph.png"
    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=dpi)

    btn = st.download_button(
        "Download figure",
        data=img,
        file_name=fn,
        mime="image/png",
        key=f"download_{linguistic}",
    )
    if btn:
        plt.close("all")

# @st.cache_data
def create_graph(
    edited_matrix,
    layout_func,
    figsize,
    nodesize,
    font_size,
    weights_font_size,
    title_font_size,
    arrowsize,
):
    fig = st.pyplot(
        create_visual_map(
            edited_matrix,
            layout_func,
            figsize,
            nodesize,
            font_size,
            weights_font_size,
            title_font_size,
            arrowsize,
        ),
        clear_figure=False,
    )


# @st.cache_data
def create_linguistic_graph(
    edited_matrix,
    layout_func,
    figsize,
    nodesize,
    font_size,
    weights_font_size,
    title_font_size,
    arrowsize,
):
    fig = st.pyplot(
        create_visual_map_linguistic(
            edited_matrix,
            layout_func,
            figsize,
            nodesize,
            font_size,
            weights_font_size,
            title_font_size,
            arrowsize,
        ),
        clear_figure=False,
    )



def create_visual_map(
    df,
    layout_func,
    figsize=10,
    node_size=1000,
    font_size=6,
    weight_font_size=6,
    title_font_size=30,
    arrowsize=10,
):
    """
    This function creates the fcm graph based on the networkx library
    """

    # todo add more colomaps and colors with a widget
    df = df.transpose()  # df is transposed due to column-wise arrows of the graph

    G = nx.MultiDiGraph()
    for i in df.columns:
        for j in df.columns:
            weight = df[i].loc[j]
            if weight != 0.0:
                G.add_edge(
                    str(i).replace(" ", "\n"), str(j).replace(" ", "\n"), weight=weight
                )

    fig, ax = plt.subplots(figsize=(figsize + figsize // 2, figsize))
    pos = layout_func(
        G
    )
    M = G.number_of_edges()
    colors = [i[2]["weight"] for i in G.edges(data=True)]

    options = {
        "node_color": "skyblue",
        "edge_color": list(colors),  # np.abs(colors)
        "width": 4,
        "edge_cmap": plt.cm.coolwarm,
        "with_labels": True,
        "node_size": node_size,
        "font_size": font_size,
        "arrows": True,
        "arrowstyle": "->",
        "arrowsize": arrowsize,
        "connectionstyle": "arc3, rad = 0.1",
        "edge_vmin": -1,
        "edge_vmax": 1,
    }

    nx.draw(G, pos, **options)
    # Add edge labels (weights)
    edge_labels = {(n1, n2): d["weight"] for n1, n2, d in G.edges(data=True)}
    labels = nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=weight_font_size,
        label_pos=0.4,
        connectionstyle="arc3, rad = 0.1",
    )  
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(-1, 1))
    sm._A = []
    
    cbar = plt.colorbar(sm, shrink=0.95, fraction=0.1, ax=ax)
    cbar.ax.tick_params(labelsize=font_size - (font_size // 4))
    plt.title("Fuzzy Cognitive Map", fontsize=title_font_size + 2)
    plt.tight_layout()  
    return fig


def create_visual_map_linguistic(
    df,
    layout_func,
    figsize=10,
    node_size=1000,
    font_size=6,
    weight_font_size=6,
    title_font_size=30,
    arrowsize=10,
):
    """
    This function creates the linguistic fcm graph based on the networkx library
    """
    # todo add more layouts with a widget
    # todo add more colomaps and colors with a widget
    df = df.transpose()  # df is transposed due to column-wise arrows of the graph

    G = nx.MultiDiGraph()
    for i in df.columns:
        for j in df.columns:
            weight = df[i].loc[j]
            if weight != "None":
                G.add_edge(i.replace(" ", "\n"), j.replace(" ", "\n"), weight=weight)

    fig, ax = plt.subplots(figsize=(figsize + figsize // 2, figsize))
    pos = layout_func(
        G
    )
    M = G.number_of_edges()
    cmap = matplotlib.colors.ListedColormap(["coral", "cyan"])
    colors = [0 if i[2]["weight"].startswith("-") else 1 for i in G.edges(data=True)]

    options = {
        "node_color": "aliceblue",
        "edge_color": list(colors),
        "width": 4,
        "edge_cmap": cmap,
        "with_labels": True,
        "node_size": node_size,
        "font_size": font_size,
        "arrows": True,
        "arrowstyle": "->",
        "arrowsize": arrowsize + 1,
        "connectionstyle": "arc3, rad = 0.1",
        "edge_vmin": 0,
        "edge_vmax": 1,
    }

    nx.draw(G, pos, ax=ax, **options)
    # Add edge labels (weights)
    edge_labels = {(n1, n2): d["weight"] for n1, n2, d in G.edges(data=True)}
    labels = nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=weight_font_size,
        label_pos=0.4,
        connectionstyle="arc3, rad = 0.1",
        ax=ax,
    ) 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm._A = []
    ticks = [0.25, 0.75]

    cbar = plt.colorbar(sm, ax=ax, ticks=ticks, shrink=0.95, fraction=0.1)
    cbar.ax.set_yticklabels(["-", "+"])
    cbar.ax.tick_params(labelsize=font_size)
    plt.title("Fuzzy Cognitive Map", fontsize=title_font_size + 1)
    plt.tight_layout() 
    return fig
