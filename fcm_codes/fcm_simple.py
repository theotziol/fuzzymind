import numpy as np
import pandas as pd 
import streamlit as st
from matplotlib import pyplot as plt


help_weight_matrix = 'Give a weight value in [-1,1]'

def kosko():
    pass

def stylios():
    pass

def rescaled():
    pass

#activation functions

def sigmoid():
    pass

def bivalent():
    pass

def trivalent():
    pass

def tanh():
    pass


dict_rules = {
    'Kosko' : kosko,
    'Stylios' : stylios,
    'Rescaled' : rescaled,
}

dict_functions = {
    'Sigmoid' : sigmoid,
    'Bivalent' : bivalent,
    'Trivalent' : trivalent, 
    'Tanh' : tanh,
}

def create_weight_matrix_columns(num_concepts : int):
    '''
    This function creates an one row df to be used with st.data_editor. It aims to allow the user to edit the concept names.
    Returns:
        -pd.DataFrame
    '''
    dic = { }
    for i in range(num_concepts):
        dic[f'concept_{i+1}'] = [f'Give a name to concept {i+1}']
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
    k = 0.25,
    node_size = 800,
    font_size = 6,
    title_font_size = 30,
    ):
    import networkx as nx
    G = nx.Graph()
    for i in df.columns:
        for j in df.columns:
            weight = df[i].loc[j]
            if weight != 0.0:
                G.add_edge(i, j, weight = weight)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    pos = nx.spring_layout(G, k=k)  # Define layout with reduced 'k' to separate nodes
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='skyblue', font_size=font_size)  # Adjust node_size and font_size
    # Add edge labels (weights)
    edge_labels = {(n1, n2): d['weight'] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)  # Adjust font_size for edge labels
    plt.title('Fuzzy Cognitive Map', )
    plt.tight_layout()  # Ensure tight layout
    return fig
    





    

