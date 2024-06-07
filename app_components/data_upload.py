import streamlit as st 
import pandas as pd 
import numpy as np 
import io


helps = {
    
    'csv' : "Upload a dataset in the form of '.csv' file",

    'Missing values' : 'Additional strings to recognize as NA/NaN.\
        \nBy default the following values are interpreted as NaN:\
        \n “ “, “#N/A”, “#N/A N/A”, “#NA”, “-1.#IND”, “-1.#QNAN”, “-NaN”, “-nan”, “1.#IND”, “1.#QNAN”, “<NA>”, “N/A”, “NA”, “NULL”, “NaN”, “None”, “n/a”, “nan”, “null “.',

    'import' : 'Import dataset with the current parameters',
    'datetime' : 'Datetime index allows for better visualization of time series data, using time attributes (hour, day, etc).',

}


def upload_widgets(): 
    csv = st.file_uploader('Upload a dataset as ".csv" file', type = 'csv', help=helps['csv'])
    if csv is not None:
        with st.expander('CSV options', not st.session_state.uploaded):
            # st.subheader('CSV options', divider='gray')
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

            cl1, cl2, cl3 = st.columns([0.4, 0.4, 0.2])
            with cl3:
                with st.popover('Advanced options'):
                    combine_index_cols = st.multiselect('Select multiple datetime columns', df.columns)
                    if len(combine_index_cols) > 0:
                        submit_parse_dates = st.checkbox('Apply', key = 'date_columns')
                        if st.session_state['date_columns']:
                            df = read_csv(csv, delimiter = delimiter, decimal = decimal, na_values = na_values_list, parse_dates=[combine_index_cols])

            st.caption(f'{csv.name} dataset')
            st.dataframe(df)

            show_info = st.toggle('Show dataset info')
            if show_info:
                c1, c2 = st.columns(2)
                with c1:
                    st.write(df.describe())
                with c2:
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text(s)
            
            col1,col2,col3 = st.columns((0.35, 0.3, 0.35))
            
            with col2:
                imported = st.button("Import data", key = 'imported', help=helps['import'], on_click=upload_callback, args = (df, ))
            
            return csv
    else: return csv
        

                        
def modify_dataset():
    st.subheader('Dataset parameters')
    df_processed = st.session_state.working_df.copy()
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.popover('Change column names'):
            st.caption("Provide the desired name for each column.")
            df_columns = pd.DataFrame(df_processed.columns, index = df_processed.columns, columns = ['New name'])
            edited_columns = st.data_editor(df_columns).to_numpy()
            edited_columns = edited_columns.reshape((edited_columns.shape[0]))
            submit_names = st.button('Submit', key = 'Submit columns', on_click=submit_columns_callback, args = (edited_columns,))
            
    with col2:
        with st.popover('Change index'):
            index_column = st.selectbox('Select the index column', df_processed.columns, index = None)
            if index_column != None:
                submit_index = st.button('Submit', key = 'submit index', on_click=submit_index_callback, args = (index_column,))
            convert_to_datetime = st.button('Convert current index to datetime', key = 'convert_datetime', on_click=to_datetime_index_callback, help=helps['datetime'])

            
                
    with col3:
        with st.popover('Delete columns'):
            to_delete = st.multiselect('Select column(s)', df_processed.columns)
            if len(to_delete) > 0:
                st.write(f'You selected {to_delete} to be deleted!')
                submit_deletion = st.button('Submit', key = 'submit_deletion',on_click=submit_deletion_callback, args = (to_delete, ) )

                
    show_dataframe = st.toggle('Show dataset')
    if show_dataframe:
        st.dataframe(df_processed)
    if st.session_state.changed:
        restore_button = st.button('Restore changes', key = 'restore', on_click=restore_df_changes_callback)
    

                
def plot_widgets():
    '''
    This functions contains the widgets to plot the imported dataset
    '''
    st.subheader('Data visualization', divider = 'blue')
    not_plotting = ('object', 'bool')
    columns = [col for col in st.session_state.working_df.columns if col not in not_plotting]
    
    column = st.selectbox('Select a single column to plot', columns, None)
    
    if column is not None:
        chart_type = st.selectbox('Select the chart type', ['Line', 'Area', 'Bar'])
        plot_column(column, chart_type)  

        

    
@st.cache_data
def plot_column(column, chart_type):
    try:
        st.caption(f"{column} chart")
        if chart_type == 'Line':
            st.line_chart(st.session_state.working_df, y = column)
        elif chart_type == 'Area':
            st.area_chart(st.session_state.working_df, y = column)
        else:
            st.bar_chart(st.session_state.working_df, y = column)
    except Exception as e:
        st.warning(f"{e}", icon="⚠️")


                    
def upload_callback(dataframe):
    '''
    stores in session 2 dataframes. The initial_df will be in background to restore the edits whereas the working_df is where all the processing occurs 
    '''
    st.session_state.initial_df = dataframe.copy()
    st.session_state.working_df = dataframe.copy()
    st.session_state.uploaded = True
    st.session_state.changed = False

def submit_columns_callback(columns):
    st.session_state.working_df.columns = columns
    st.session_state.changed = True
    st.success('Changes applied')

def submit_index_callback(index_column):
    st.session_state.working_df.set_index(index_column, inplace = True)
    st.session_state.changed = True
    st.success('Changes applied')

def submit_deletion_callback(to_delete):
    st.session_state.working_df.drop(to_delete, inplace = True, axis = 1)
    st.session_state.changed = True
    st.success('Changes applied')

def restore_df_changes_callback():
    st.session_state.working_df = st.session_state.initial_df.copy()
    st.session_state.changed = False
    st.success('Changes discarded')



def to_datetime_index_callback():
    try:
        st.session_state.working_df.index = pd.to_datetime(st.session_state.working_df.index)
        st.session_state.changed = True
        st.success('Index is succesfully converted to datetime')
    except Exception as e:
        st.warning(f'{e}', icon="⚠️")


@st.cache_data
def read_csv(csv, **kwargs):
    return pd.read_csv(csv, **kwargs)





            
               



            






    