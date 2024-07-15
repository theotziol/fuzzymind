import streamlit as st 
import pandas as pd 
import numpy as np 


min_max_formula = '$x_i^{new} = \\dfrac{x_i^{old} - \\textbf{x}_{min}}{\\textbf{x}_{max} - \\textbf{x}_{min}}$'
standard_scaler_formula = '$x_i^{new} = \\dfrac{x_i^{old} - \\textbf{x}_{mean}}{\\textbf{x}_{std}}$'

def data_normalization():
    st.subheader('Data Normalization', divider = 'gray')
    text_columns = check_text(st.session_state.working_df)
    if len(text_columns) > 0 :
        st.warning(f'The working dataset contains the following non-numeric columns: \n{text_columns}', icon="⚠️")
        st.info('Use the **Data Transformation** tab to modify these columns, or the **Dataset Parameters** tab to delete columns.')
    else:
        col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
        with col1:
            method = st.selectbox('Select the normalization method', ['Min-Max Normalization', 'Standard Scaler'], 0 )

        with col3:
            if method == 'Min-Max Normalization':
                st.write('Rescales values to [0, 1] with the following formula:')
                st.write(min_max_formula)
                
            elif method == 'Standard Scaler':
                st.write('The z-score method (often called standardization) transforms each column into distribution with a mean of 0 and a typical deviation of 1. The following formula is applied:')
                st.write(standard_scaler_formula)
                
                
        if method == 'Min-Max Normalization':
            apply_changes = st.button('Apply normalization', on_click=apply_min_max_callback)
            if apply_changes:
                st.success('Dataset was succesfully normalized.')
        elif method == 'Standard Scaler':
            #warn and disable as it has not been tested with standarization
            st.warning('This method is not compatible with the FCM theory that expects FCM concept values to be in [0, 1]. Please select the Min-Max method for consistency!')
            apply_changes = st.button('Apply normalization', on_click=apply_standard_callback, disabled=True)
            if apply_changes:
                st.success('Dataset was succesfully normalized.')



@st.cache_data
def check_text(df):
    '''
    checks if a column with text exists, as it will raise an exception when normalization is applied in text columns
    '''
    text_dtypes = ('object', 'bool', 'datetime64[ns]')
    columns = [
        col for col in st.session_state.working_df.columns if st.session_state.working_df[col].dtype.name in text_dtypes
        or np.issubdtype(st.session_state.working_df[col].dtype, np.datetime64) 
        or not np.issubdtype(st.session_state.working_df[col].dtype, np.number)
            ]

    return columns
    


### submit callbacks

def apply_min_max_callback():
    st.session_state.non_norm_working_df = st.session_state.working_df.copy()
    for column in st.session_state.working_df.columns:
        st.session_state.working_df[column] = (st.session_state.working_df[column] - st.session_state.working_df[column].min()) / (st.session_state.working_df[column].max() - st.session_state.working_df[column].min())
    st.session_state.changed = True
    st.session_state.normalized = True

def apply_standard_callback():
    for column in st.session_state.working_df.columns:
        st.session_state.working_df[column] = (st.session_state.working_df[column] - st.session_state.working_df[column].mean()) / (st.session_state.working_df[column].std())
    st.session_state.changed = True
    

    