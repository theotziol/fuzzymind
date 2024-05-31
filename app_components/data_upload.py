import streamlit as st 
import pandas as pd 
import numpy as np 


helps = {
    
    'csv' : "Upload a dataset in the form of '.csv' file",

    'Missing values' : 'Additional strings to recognize as NA/NaN.\
        \nBy default the following values are interpreted as NaN:\
        \n “ “, “#N/A”, “#N/A N/A”, “#NA”, “-1.#IND”, “-1.#QNAN”, “-NaN”, “-nan”, “1.#IND”, “1.#QNAN”, “<NA>”, “N/A”, “NA”, “NULL”, “NaN”, “None”, “n/a”, “nan”, “null “.',
    'datetime' : 'Activate this widget if the two or more columns contain date and time information. Datetime columns are primarily used in timeseries data and contain information regarding the date and the time of the measurement. It is recommended to be used as index column'
}


def upload_widgets(): 
    csv = st.file_uploader('Upload a dataset as ".csv" file', type = 'csv', help=helps['csv'])
    if csv is not None:
        st.subheader('CSV options', divider='gray')
        col1, col2, col3 = st.columns([0.25, 0.25, 0.5])
        with col1:
            delimiter = st.radio("Select file's delimiter", [',', '.', ';'],
                        index = 0,
                        captions = ['Comma', 'Period', 'Semicolon'])
        with col2:
            decimal = st.radio("Select file's decimal", [',', '.'],
                                index = 1,
                                captions = ['Comma', 'Period'])
        with col3:
            na_values_list = st.multiselect('Define additional symbols for missing values', ['*', '#', '$', 'Other'], help=helps['Missing values'])
            if 'Other' in na_values_list:
                additional_nans = st.text_area("Provide text for nan values, you may use **double comma ',,' for multiple insertions. Do not use space ' ' before or after ,, insertions**")
                if additional_nans is not None:
                    additional_nans = additional_nans.split(',,')
                    na_values_list.remove('Other')
                    for i in additional_nans:
                        na_values_list.append(i)
                    st.text(f'Additional provided missining value symbols: {na_values_list}')
        
       
        
        df = read_csv(csv, delimiter = delimiter, decimal = decimal, na_values = na_values_list)
        show_dataframe = st.toggle('Show dataset')
        if show_dataframe:
            st.caption('Uploaded dataset')
            st.dataframe(df)
        
        return csv,df 
    else: return csv, None
        

                        
def modify_dataset(df_processed):
    with st.expander('Modify dataset...'):
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.popover('Change column names'):
                st.caption("Provide the desired name for each column.")
                df_columns = pd.DataFrame(df_processed.columns, index = df_processed.columns, columns = ['New name'])
                edited_columns = st.data_editor(df_columns).to_numpy()
                edited_columns = edited_columns.reshape((edited_columns.shape[0]))
                submit_names = st.button('Submit', key = 'Submit columns')
                if submit_names:
                    df_processed.columns = edited_columns
                    st.success('Changes applied')
        with col2:
            with st.popover('Change index'):
                index_column = st.selectbox('Select the index column', df_processed.columns, index = None)
                if index_column != None:
                    submit_index = st.button('Submit', key = 'submit_index')
                    if submit_index:
                        df_processed.set_index(index_column, inplace = True)
                        st.success('Changes applied')
        with col3:
            with st.popover('Delete columns'):
                to_delete = st.multiselect('Select column(s)', df_processed.columns)
                if len(to_delete) > 0:
                    st.write(f'You selected {to_delete} to be deleted!')
                    submit_deletion = st.button('Submit', key = 'submit_deletion')
                    if submit_deletion:
                        df_processed.drop(to_delete, axis =1, inplace = True)
                        st.success('Changes applied')
        show_dataframe = st.toggle('Show dataset')
        if show_dataframe:
            st.dataframe(df_processed)
    
    return df_processed

                



                    
                    


@st.cache_data
def read_csv(csv, **kwargs):
    return pd.read_csv(csv, **kwargs)





            
               



            






    