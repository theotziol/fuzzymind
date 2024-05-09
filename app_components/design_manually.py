import streamlit as st
import sys

sys.path.insert(1, '../fcm_codes')
from fcm_codes.general_functions import *

def manual_tab():
    st.subheader('Define the total number of concepts', divider = 'green')
    num_concepts = st.number_input('Give the number of concepts', min_value=3, max_value=50, value = None, help = 'Give an integer in the range [3, 50]')
    if num_concepts != None:
        st.subheader('Define concepts', divider = 'green')
        columns_df = create_weight_matrix_columns(num_concepts)
        edited_columns = st.data_editor(columns_df, hide_index=True)
        st.subheader('Define weighted interconnections', divider = 'green')
        weight_matrix_df = create_weight_matrix(num_concepts, edited_columns.values.tolist())
        
        edited_matrix = st.data_editor(weight_matrix_df.style.apply(highlight_diagonal, axis=None), hide_index=True, disabled = ['-'], column_config=fix_configs(weight_matrix_df))
        edited_matrix.set_index('-', inplace = True)
        edited_matrix = edited_matrix.astype(float)
        return edited_matrix, True
    else:
        return None, False