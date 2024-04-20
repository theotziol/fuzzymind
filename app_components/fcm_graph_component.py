import streamlit as st 
from matplotlib import pyplot as plt
import sys

sys.path.insert(1, '../fcm_codes')
from fcm_codes.fcm_simple import *
import io 

help_k = 'This is a parameter that is used to calculate the distance of the concepts'
help_DPI = 'The DPI value for saving the graph'
def graph(edited_matrix):
    col1, col2 = st.columns(2, gap = 'small')
    with col1:
        #this column is for modifying the graph parameters
        figsize = st.slider('Figure size', 5, 20, 10, 1)
        k = st.slider('Parameter k', 0.1, 0.9, 0.3, 0.1, help=help_k)
        nodesize = st.slider('Node size', 100, 2000, 800, 100)
        font_size = st.slider('Font size', 3, 30, 10, 1)
        title_font_size = st.slider('Font size (title)', 10, 50, 30, 1)
        
    with col2:
        #this column is for the graph
        st.pyplot(create_visual_map(edited_matrix, figsize, k, nodesize, font_size, title_font_size))
    
    dpi = st.slider('DPI', 300, 1500, 600, 100, help=help_DPI)
    fn = 'FCM_graph.png'
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi = dpi)

    btn = st.download_button('Download figure', data = img, file_name=fn, mime="image/png")

