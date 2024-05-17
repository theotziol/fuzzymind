#this file contains functions for back-end construction of the FCM and the graphs

import numpy as np
import pandas as pd 
import streamlit as st
from matplotlib import pyplot as plt
import matplotlib as mpl


help_weight_matrix = 'Give a weight value in [-1,1]'
help_weight_matrix_linguistic = 'Double click on each cell to select the fuzzy relationship among concepts based on the defined mfs'
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

def create_linguistic_weight_matrix(num_concepts:int, concept_columns : list, mfs : list):
    '''
    it creates a weight matrix with shape (num_concepts, num_concepts+1). 
    The reason behind this is that st.data_editor does not allow multiindex (str index), thus the concept names cannot be passed as index.
    To resolve this, a new column is inserted in the begining of the dataframe (column_name = '-') that contains the concept names.
    Args:
        num_concepts : int, the number of concepts
        concept_columns : list, the concept names
        mfs: list, the defined memberships functions

    Returns:
        str type pd.DataFrame
    '''

    columns = np.array(concept_columns)
    value = mfs[(len(mfs)//2)] #take the middle as initial value (expected to be None)
    matrix = np.full((num_concepts, num_concepts+1), value, dtype="object")
    
    matrix[:, 0] = columns
    columns = np.insert(columns, 0, '-')
    df = pd.DataFrame(matrix, columns=columns)
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

def fix_configs_linguistic(df, mfs, index = '-'):
    '''
    This function employs the st.column_config to ensure that each weight that is passed is from the fuzzy memberships that were previously defined
    Args: 
        df :  The pd.DataFrame that will be given to the st.data_editor,
        mfs : list, the defined membership functions,
        index : The dummy column name that is given as index. default ('-')
    '''
    config = {

    }
    for column in df.columns:
        if column == index:
            continue
        else:
            config[column] = st.column_config.SelectboxColumn(help = help_weight_matrix_linguistic, options = mfs)
    return config


# Define a custom styling function to highlight the diagonal
def highlight_diagonal(x):
    # Create an empty DataFrame of the same shape as x
    style_df = pd.DataFrame('', index=x.index, columns=x.columns)
    # Highlight the diagonal elements
    np.fill_diagonal(style_df.values[:, 1:], 'background-color: gray')
    return style_df



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



    

