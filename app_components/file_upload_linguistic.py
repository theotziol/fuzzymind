import streamlit as st 
import pandas as pd 
import json
from app_components.design_manually_linguistic import *

load_widget_help = "Upload a '.csv' file.\n\
    It is recommended to upload a '.csv' file where both the first row (header) and the first column (index) contain the concepts' notation/names.\n\
        You can use [Design Manually] option to construct and download manually a weight matrix. "

load_widget_json_help = "Upload a '.json' file.\n\
    A '.json' file is required which contains information on how to create the fuzzy membership values. \n\
        Use the tab define manually -> Parameters to define and download a json file. \n\
        The json must have:\n\
        key1 = 'method' : This describe the membership type and accepts values ['Triangular', 'Trapezoidal', 'Gaussian']\n\
        key2 = 'range' : the universe of discource with value [-1.0, 1.0]\n\
        key3 = 'step' : 0.01, this value indicates the decimals when defining Universe of discource\n\
        key4 = 'memberships: and as value is a dictionairy (key-value pairs again), where the names and the attributes of the mfs are passed.\n\
        e.g.  None :[-0.25, 0, 0.25] where None the name, -0.25 the beggining of the triangle, 0 the μ = 1, and 0.25 the end of the triangle"

index_boolean_widget_help = "The uploaded file is expected to has concepts' names/notation as index (first column)\n\
    Set this toggle widget ON if the first column in the .csv file has a separate index\n\
        i.e. first column = [0,1,2,..., n] and second column = [c1, c2, ..., cn]."

def matrix_upload():
    '''
    The component for loading a .csv file
    Returns:
        file: type = object. object attributes : [file_id, name, type, size, upload_url, delete_url] i.e. file.name returns the name of the file
        dataframe: pd.DataFrame of the weight matrix
    '''
    col1, col2 = st.columns(2, gap = 'medium')
    with col1:
        weight_matrix_file = st.file_uploader('Upload a .csv file', type = 'csv', help=load_widget_help)
        fuzzy_variables_file = st.file_uploader('Upload a .json file of the membership functions', type = 'json', help=load_widget_json_help)
    if weight_matrix_file is not None:
        with col2:
            delimiter = st.radio("Select file's delimiter", [',', '.', ';'],
                                index = 0,
                                captions = ['Comma (default)', 'Full Stop (dot)', 'Semicolon'], horizontal = True)
            decimal = st.radio("Select file's decimal", [',', '.'],
                                index = 1,
                                captions = ['Comma', 'Dot (default)'], horizontal = True)
            boolean_index = st.toggle('Contains index', help=index_boolean_widget_help)
            if boolean_index:
                index_col = 1
            else:
                index_col = 0
            dataframe = pd.read_csv(weight_matrix_file, delimiter=delimiter, decimal = decimal, index_col=index_col, dtype = 'object').fillna('None')
        st.write(dataframe)
    else:
        dataframe = None
    
    if fuzzy_variables_file is not None:
        dic = json.load(fuzzy_variables_file)
        with st.expander('Modify mfs parameters...'):
            final_dic = modify_fuzzy_memberships(dic)
    else:
        st.write('Pass a .json file')
        final_dic = None

    return dataframe, weight_matrix_file, final_dic