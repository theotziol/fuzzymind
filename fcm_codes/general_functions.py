#this file contains functions for back-end construction of the FCM and the graphs

import numpy as np
import pandas as pd 
import streamlit as st
from matplotlib import pyplot as plt
import matplotlib as mpl


help_weight_matrix = 'Give a weight value in [-1,1]'
help_state_vector_sigmoid = 'Give a concept value in [0,1]'
help_state_vector_tanh = 'Give a concept value in [-1,1]'
help_state_vector_bivalent = 'Give a concept value in {0,1}'
help_state_vector_trivalent = 'Give a concept value in {-1,0,1}'


rules = [
    'Kosko' ,
    'Modified Kosko',
    'Rescaled',
]

functions = [
    'Sigmoid' ,
    'Bivalent' ,
    'Trivalent' , 
    'Tanh',
]

def create_weight_matrix_columns(num_concepts : int):
    '''
    This function creates an one row df to be used with st.data_editor. It aims to allow the user to edit the concept names.
    Returns:
        -pd.DataFrame
    '''
    dic = { }
    for i in range(num_concepts):
        dic[f'concept_{i+1}'] = [f'name of concept {i+1}']
    df = pd.DataFrame(dic)
    return df

def create_weight_matrix(num_concepts:int, concept_columns : list):
    '''
    it creates a weight matrix with shape (num_concepts, num_concepts+1). 
    The reason behind this is that st.data_editor does not allow multiindex (str index), thus the concept names cannot be passed as index.
    To resolve this, a new column is inserted in the begining of the dataframe (column_name = '-') that contains the concept names.
    Args:
        num_concepts : int, the number of concepts
        concept_columns : list, the concept names

    Returns:
        str type pd.DataFrame
    '''

    columns = np.array(concept_columns)
    zeros = np.zeros((num_concepts, num_concepts+1))

    zeros = zeros.astype(str)
    zeros[:, 0] = columns
    columns = np.insert(columns, 0, '-')
    df = pd.DataFrame(zeros, columns=columns)
    return df

def fix_configs(df, index = '-'):
    '''
    This function employs the st.column_config to ensure that each weight that is passed is in [-1, 1]
    Args: 
        df :  The pd.DataFrame that will be given to the st.data_editor
        index : The dummy column name that is given as index. default ('-')
    '''
    config = {

    }
    for column in df.columns:
        if column == index:
            continue
        else:
            config[column] = st.column_config.NumberColumn(help = help_weight_matrix, min_value = -1.0, max_value = 1.0, step = 0.01)
    return config


# Define a custom styling function to highlight the diagonal
def highlight_diagonal(x):
    # Create an empty DataFrame of the same shape as x
    style_df = pd.DataFrame('', index=x.index, columns=x.columns)
    # Highlight the diagonal elements
    np.fill_diagonal(style_df.values[:, 1:], 'background-color: gray')
    return style_df


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
    import networkx as nx
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
    
    cbar = plt.colorbar(sm, ticks = ticks, shrink = 0.95, fraction = 0.1)
    cbar.ax.set_yticklabels([f'$\\pm {i}$' if i != 0 else '0' for i in ticks])
    cbar.ax.tick_params(labelsize = font_size)
    plt.title('Fuzzy Cognitive Map', fontsize = title_font_size + 1)
    plt.tight_layout()  # Ensure tight layout
    return fig
    





def create_initial_state_vector(concepts, activation):
    '''
    This function creates an one row df to be used with st.data_editor. It aims to allow the user to edit the concept initial values.
    Args:
        concept names
    Returns:
        -pd.DataFrame
        -config dic: The file that 
    '''
    config = {}
    array = np.zeros((1, len(concepts)))
    df = pd.DataFrame(array, columns=concepts)

    if activation == 'Sigmoid':
        help_text = help_state_vector_sigmoid
        min_value, max_value, step = 0, 1, 0.01
    elif activation == 'Tanh':
        help_text = help_state_vector_tanh
        min_value, max_value, step = -1, 1, 0.01
    elif activation == 'Bivalent':
        help_text = help_state_vector_bivalent
        min_value, max_value, step = 0, 1, 1
    elif activation == 'Trivalent':
        help_text = help_state_vector_trivalent
        min_value, max_value, step = -1, 1, 1
    else:
        help_text = help_state_vector_tanh
        min_value, max_value, step = -1, 1, 0.01

    for column in df.columns:
        config[column] = st.column_config.NumberColumn(help = help_text, min_value = min_value, max_value = max_value, step = step)
    return df, config



    

