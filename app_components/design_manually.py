import streamlit as st
import sys

sys.path.insert(1, '../fcm_codes')
from fcm_codes.general_functions import *

dic_variables_caption = {
    5 : ':red[-High], :red[-Low],  None,  :blue[+Low], :blue[+High]',
    7 : ':red[-High], :red[-Medium], :red[-Low],  None,  :blue[+Low], :blue[+Medium], :blue[+High]',
    11 : ':red[-Very High], :red[-High], :red[-Medium], :red[-Low], :red[-Very Low],  None,  :blue[+Very Low] :blue[+Low], :blue[+Medium], :blue[+High], :blue[+Very High],',
}

dic_variables = {
    5 : ['-High', '-Low', 'None', '+Low', '+High'],
    7 : ['-High','-Medium', '-Low', 'None', '+Low', '+Medium', '+High'],
    11 : ['-Very High','-High','-Medium', '-Low', '-Very Low', 'None', '+Very Low', '+Low', '+Medium', '+High']
}


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
    
def fuzzy_sets():
    '''
    The streamlit widgets for creating fuzzy sets
    '''
    st.subheader('Define fuzzy sets', divider = 'green')
    fuzzy_variables = st.radio('Select fuzzy variables', [5, 7, 11], captions=[dic_variables_caption[i] for i in dic_variables_caption.keys()])
    with st.expander('Parameters...'):
        membership = st.selectbox('Select Membership Function', ['Triangular', 'Trapezoidal', 'Gaussian'], index = 0)
        st.write('$\\mathbb{U} = [-1, 1]$')
        col1, col2 = st.columns(2)
        with col1:
            #this column is for memberhsip's parameters modification
            st.write("Modify Membership Parameters")
            
        with col2:
            #this column is for plotting the fuzzy sets 
            pass



def manual_tab_linguistic():
    '''
    The main tab for manual linguistic fcm construction
    '''
    st.subheader('Define the total number of concepts', divider = 'green')
    num_concepts = st.number_input('Give the number of concepts', min_value=3, max_value=50, value = None, help = 'Give an integer in the range [3, 50]')
    if num_concepts != None:
        st.subheader('Define concepts', divider = 'green')
        columns_df = create_weight_matrix_columns(num_concepts)
        edited_columns = st.data_editor(columns_df, hide_index=True)
        
        

        # weight_matrix_df = create_weight_matrix(num_concepts, edited_columns.values.tolist())
        
        # edited_matrix = st.data_editor(weight_matrix_df.style.apply(highlight_diagonal, axis=None), hide_index=True, disabled = ['-'], column_config=fix_configs(weight_matrix_df))
        # edited_matrix.set_index('-', inplace = True)
        # edited_matrix = edited_matrix.astype(float)
        
    #     return edited_matrix, True
    # else:
    #     return None, False