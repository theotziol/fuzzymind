import streamlit as st 
import pandas as pd 
import json
from app_components.design_manually_linguistic import *
from copy import deepcopy as dc
import io 


## Basic info text variables
basic_info = "Upload pairs of '.csv'/'.json' files with the same name. E.g. **'file1.csv'** and **'file1.json'.**"

compatibility_info = "Currently this app is designed to work with weight matrices and json files that derived from this app. Use the Design Manually tab to construct and download such files."

load_widget_help = "It is recommended to upload a '.csv' file where both the first row (header) and the first column (index) contain the concepts' notation/names.\n\
        You can use [Design Manually] option to construct and download manually a weight matrix.\n\
        **Currently the weight matrices that are supported are based on the parameters that are used with the Design Manually tab.**"

load_widget_help_multiple = "It is recommended to upload '.csv' files where both the first row (header) and the first column (index) contain the concepts' notation/names.\n\
        You can use [Design Manually] option to construct and download manually a weight matrix. "

load_widget_json_help = "'.json' files are required that contain information on how to create the fuzzy membership values. \n\
        **Use the tab define manually -> Parameters to define and download a json file.**\n\n\
        The json must have:\n\n\
        key1 = 'method' : This describe the membership type and accepts values ['Triangular', 'Trapezoidal', 'Gaussian']\n\n\
        key2 = 'range' : the universe of discource with value [-1.0, 1.0]\n\n\
        key3 = 'step' : 0.01, this value indicates the decimals when defining Universe of discource\n\n\
        key4 = 'memberships: and as value is a dictionairy (key-value pairs again), where the names and the attributes of the mfs are passed.\n\
        e.g.  None :[-0.25, 0, 0.25] where None the name, -0.25 the beggining of the triangle, 0 the μ = 1 (head of triangle), and 0.25 the end of the triangle."

index_boolean_widget_help = "The uploaded file is expected to has concepts' names/notation as index (first column).\n\
    Set this toggle widget ON if the first column in the .csv file has a separate index\n\
        i.e. first column = [0,1,2,..., n] and second column = [c1, c2, ..., cn]."



### functions for file uploading
def matrix_upload():
    '''
    The component for loading a .csv and . json file
    Returns:
        file: type = object. object attributes : [file_id, name, type, size, upload_url, delete_url] i.e. file.name returns the name of the file
        dataframe: pd.DataFrame of the weight matrix
    '''
    st.info(compatibility_info)
    with st.expander('Upload files...'):
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
            st.caption('Uploaded linguistic weight matrix')
            st.write(dataframe)
        else:
            dataframe = None
        
    if fuzzy_variables_file is not None:
        dic = json.load(fuzzy_variables_file)
        with st.expander('Modify mfs parameters...'):
            final_dic = modify_fuzzy_memberships(dic)
    else:
        st.write("No '.json' file is uploaded")
        final_dic = None

    return dataframe, weight_matrix_file, final_dic


def matrices_upload():
    '''
    The component for loading multiple .csv and json files. It provides the widgets for uploading and utilizes two function for checking and handling the raw data.
    Returns:
        
    '''
    st.info(compatibility_info)
    with st.expander('Upload files...'):
        st.info(basic_info)
        col1, col2 = st.columns(2, gap = 'medium')
        with col1:
            weight_matrix_files = st.file_uploader('Upload a .csv file', type = 'csv', help=load_widget_help, accept_multiple_files=True)
            fuzzy_variables_files = st.file_uploader('Upload a .json file of the membership functions', type = 'json', help=load_widget_json_help, accept_multiple_files=True)
        if len(weight_matrix_files) > 0 and len(fuzzy_variables_files) > 0:
            pairs_boolean = check_uploads(weight_matrix_files, fuzzy_variables_files)
            with col2:
                delimiter = st.radio("Select files' delimiter", [',', '.', ';'],
                                    index = 0,
                                    captions = ['Comma (default)', 'Full Stop (dot)', 'Semicolon'], horizontal = True)
                decimal = st.radio("Select files' decimal", [',', '.'],
                                    index = 1,
                                    captions = ['Comma', 'Dot (default)'], horizontal = True)
                boolean_index = st.toggle('Contains index', help=index_boolean_widget_help)
                if boolean_index:
                    index_col = 1
                else:
                    index_col = 0
            dic_pairs = handle_uploads(weight_matrix_files, fuzzy_variables_files, delimiter, decimal, index_col)
            return dic_pairs
        else:
            return None

        


def check_uploads(csv_files, json_files):
    '''
    This function checks if the uploads meet the requirements of same name same length. Raises an error if not.
    Returns True if no issue was found and False otherwise
    '''
    names_csv = []
    names_json = []

    pairs = False


    for i in csv_files:
        names_csv.append(i.name.replace('.csv', ''))
    
    for i in json_files:
        names_json.append(i.name.replace('.json', ''))

    # errors codes
    if len(csv_files) != len(json_files):
        st.error('Error, number of uploaded csv files do not match the number of uploaded json files')
        st.info("Upload equal number of files! **Notice, files should have the same name! E.g. 'file1.csv' / 'file1'.json'.**")
    else:
        missing_files_csv = []
        missing_files_json = []
        for i in names_csv:
            if i in names_json:
                continue
            else:
                missing_files_csv.append(i + '.csv')
        for i in names_json:
            if i in names_csv:
                continue
            else:
                missing_files_json.append(i + '.json')
        if len(missing_files_csv) > 0 or len(missing_files_json) > 0:
            if len(missing_files_csv) != 0:
                st.error(f'Error... File(s) {missing_files_csv} does/do not correspond to a .json file')
            if len(missing_files_json):
                st.error(f'Error... File(s) {missing_files_json} does/do not correspond to a .csv file')
            st.info(basic_info)
        else:
            st.success("Uploaded '.csv' files match uploaded '.json' files")
            pairs = True

    return pairs
        


@st.cache_data    
def handle_uploads(csv_files, json_files, delimiter, decimal, index_col):
    '''
    it reads the CSVs and jsons and combines them under a single dictionairy.
    Returns:
        __type__ dict : Keys are the filenames and values are lists with weight matrix (pd.DataFrame) nad mfs (dict)
    '''
    indexing_list_jsons = dc(json_files) #use this list to delete json files after matching; thus avoiding unnecessary repeats in loop

    dictionairy = dict.fromkeys([i.name.replace('.csv', '') for i in csv_files], None)

    for csv in csv_files:
        try:
            dataframe = pd.read_csv(csv, delimiter=delimiter, decimal = decimal, index_col=index_col, dtype = 'object').fillna('None')
        except Exception as e:
            st.error(f'Error with file {csv.name}\n\n{e}')
            continue
        for i, json_file in enumerate(indexing_list_jsons):
            try:
                dic = json.loads(json_file.getvalue())
                if csv.name.replace('.csv', '') == json_file.name.replace('.json', ''):
                    dictionairy[csv.name.replace('.csv', '')] = [dataframe, dic]
                    del indexing_list_jsons[i]
            except Exception as e:
                st.error(f'Error with file {json_file.name}\n\n{e}')
                del indexing_list_jsons[i]
                continue
    return dictionairy

                
    