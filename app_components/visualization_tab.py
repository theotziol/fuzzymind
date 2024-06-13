import streamlit as st 
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import colors



                
def plot_widgets():
    '''
    This functions contains the widgets to plot the imported dataset
    '''
    st.subheader('Data visualization', divider = 'blue')
    not_plotting = ('object', 'bool')
    columns = [col for col in st.session_state.working_df.columns if st.session_state.working_df[col].dtype.name not in not_plotting]

    chart_type = st.selectbox('Select the chart type', ['Line', 'Area', 'Bar', 'Boxplot', 'Histogram'])
    if chart_type == 'Boxplot':
        columns = st.multiselect('Select column(s) to plot', columns, None)
        if len(columns) > 0:
            to_plot = st.toggle('Generate graph')
            if to_plot:
                plot_column(columns, chart_type)         

    else:
        column = st.selectbox('Select column to plot', columns, None)
        if column is not None:
            to_plot = st.toggle('Generate graph')
            if to_plot:
                plot_column(column, chart_type)

        



def plot_column(column, chart_type):
    try:
        st.caption(f"{column} chart")
        if chart_type == 'Line':
            st.line_chart(st.session_state.working_df, y = column)
        elif chart_type == 'Area':
            st.area_chart(st.session_state.working_df, y = column)
        elif chart_type == 'Bar':
            st.bar_chart(st.session_state.working_df, y = column)
        elif chart_type == 'Boxplot':
            width = st.slider("Change graph's width", 5, 30, 8)
            height = st.slider("Change graph's height", 5, 30, 6)
            fig, axs = plt.subplots(figsize = (width, height))
            st.session_state.working_df.boxplot(column= column,ax = axs, rot = 10)
            st.pyplot(fig)

        elif chart_type == 'Histogram':
            width = st.slider("Change graph's width", 5, 30, 8)
            height = st.slider("Change graph's height", 5, 30, 6)
            bins = st.slider('Change histogram bins', 10, 100, 30)
            fig, axs = plt.subplots(figsize = (width, height))
            axs.hist(st.session_state.working_df[column], bins = bins)
            axs.set_ylabel('Cardinality')
            axs.axvline(st.session_state.working_df[column].mean(), color = 'cyan', linestyle='dashed', linewidth=2, label = 'mean')
            axs.axvline(st.session_state.working_df[column].median(), color = 'k', linestyle='dashed', linewidth=1, label = 'median')
            axs.legend()
            st.pyplot(fig)
            

    except Exception as e:
        st.warning(f"{e}", icon="⚠️")