import streamlit as st 




def warning_signs():
    '''
    Provide warning signs that would affect the learning.
    '''
    columns_na = [i for i in st.session_state.working_df.columns if st.session_state.working_df[i].isna().any()]
    text_dtypes = ('object', 'bool')
    columns = [col for col in st.session_state.working_df.columns if st.session_state.working_df[col].dtype.name in text_dtypes]
    if len(columns_na) > 0:
        st.warning('Your dataset contains NaN values. Learning will crush.', icon = "⚠️")
    if len(columns) > 0:
        st.warning('Your dataset contains columns with text. Learning will crush.', icon = "⚠️")
