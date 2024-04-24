import streamlit as st 
from matplotlib import pyplot as plt
import sys

sys.path.insert(1, '../fcm_codes')
from fcm_codes.general_functions import *
import io 

#help_label_pos = 'This is a parameter used to define the location of the label, with 0.5 at center (avoid to to ovelap)'
help_DPI = 'The DPI value of the graph'

@st.cache_data
def create_graph(edited_matrix, figsize, nodesize, font_size, weights_font_size, title_font_size, arrowsize,):
    fig = st.pyplot(create_visual_map(edited_matrix, figsize, nodesize, font_size, weights_font_size, title_font_size, arrowsize), clear_figure=False)

def graph(edited_matrix):
    st.subheader('FCM graph')
    col1, col2 = st.columns(2, gap = 'small')
    with col1:
        #this column is for modifying the graph parameters
        figsize = st.slider('Figure size', 5, 20, 10, 1)
        nodesize = st.slider('Node size', 300, 4000, 1500, 100)

        # check if needed
        # label_pos = st.slider('Node size', 0.1, 0.9, 0.4, 0.1) 
        # font_size = st.slider('Font size (nodes)', 3, 40, 10, 1)
        # weights_font_size = st.slider('Font size (weights)', 5, 30, 10, 1)
        # title_font_size = st.slider('Font size (title)', 10, 100, 30, 5)

        dpi = st.slider('DPI', 300, 1500, 600, 100, help=help_DPI)
        font_size = nodesize // 200
        weights_font_size = font_size
        arrowsize = weights_font_size + 1 
        title_font_size = figsize*2
        
        
    with col2:
        #this column is for the graph
        fig = create_graph(edited_matrix, figsize, nodesize, font_size, weights_font_size, title_font_size, arrowsize)
    
    
    fn = 'FCM_graph.png'
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi = dpi)

    btn = st.download_button('Download figure', data = img, file_name=fn, mime="image/png")
    if btn:
        plt.close('all')

