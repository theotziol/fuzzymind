import streamlit as st 
from matplotlib import pyplot as plt
import sys

sys.path.insert(1, '../fcm_codes')
from fcm_codes.general_functions import *
import io 

help_k = 'This is a parameter used to calculate the distance of the concepts'
help_DPI = 'The DPI value of the graph'

@st.cache_data
def create_graph(edited_matrix, figsize, k, nodesize, font_size, weights_font_size, title_font_size):
    fig = st.pyplot(create_visual_map(edited_matrix, figsize, k, nodesize, font_size, weights_font_size, title_font_size), clear_figure=False)

def graph(edited_matrix):
    st.subheader('FCM graph')
    col1, col2 = st.columns(2, gap = 'small')
    with col1:
        #this column is for modifying the graph parameters
        figsize = st.slider('Figure size', 5, 20, 10, 1)
        k = st.slider('Parameter k', 0.1, 0.9, 0.3, 0.1, help=help_k)
        nodesize = st.slider('Node size', 200, 2000, 1000, 100)
        font_size = st.slider('Font size (nodes)', 3, 40, 10, 1)
        weights_font_size = st.slider('Font size (weights)', 5, 30, 10, 1)
        title_font_size = st.slider('Font size (title)', 10, 100, 30, 5)
        
    with col2:
        #this column is for the graph
        fig = create_graph(edited_matrix, figsize, k, nodesize, font_size, weights_font_size, title_font_size)
    
    dpi = st.slider('DPI', 300, 1500, 600, 100, help=help_DPI)
    fn = 'FCM_graph.png'
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi = dpi)

    btn = st.download_button('Download figure', data = img, file_name=fn, mime="image/png")
    if btn:
        plt.close('all')

