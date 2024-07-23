import streamlit as st 
import pandas as pd 
import sys 

sys.path.insert(1, '../fcm_codes')
from fcm_codes.general_functions import *

load_widget_help = "Upload a '.csv' file.\n\
    It is recommended to upload a '.csv' file where both the first row (header) and the first column (index) contain the concepts' notation/names.\n\
        You can use [Design Manually] option to construct and download manually a weight matrix. "

index_boolean_widget_help = "The uploaded file is expected to has concepts' names/notation as index (first column)\n\
    Set this toggle widget ON if the first column in the .csv file has a separate index\n\
        i.e. first column = [0,1,2,..., n] and second column = [c1, c2, ..., cn]."

help_edit_matrix = "Activating this widget will provide a new table-widget where the weight values (cells) can be modified."

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
            dataframe = pd.read_csv(weight_matrix_file, delimiter=delimiter, decimal = decimal, index_col=index_col)
            dataframe = dataframe.astype(float)
        st.caption('Uploaded weight matrix.')
        st.write(dataframe)
        edit = st.toggle("Edit weight matrix values", False, help = help_edit_matrix)
        if edit:
            st.caption('Edited weight matrix.')
            dataframe = st.data_editor(dataframe, disabled = ['-'], column_config=fix_configs(dataframe))
            dataframe = dataframe.astype(float)

    else:
        dataframe = None
    
    return dataframe, weight_matrix_file
        
        
    