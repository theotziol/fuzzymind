import streamlit as st
import pandas as pd
import numpy as np
import sys
from copy import deepcopy as dc

sys.path.insert(1, "../fcm_codes")
from fcm_codes.general_functions import *


def manual_tab():
    st.subheader("Define the total number of concepts", divider="green")
    
    # ADDED KEY: "num_concepts_widget" lets Streamlit handle the widget's internal state
    num_concepts = st.number_input(
        "Give the number of concepts",
        min_value=3,
        max_value=50,
        value=st.session_state.get('num_concepts', None),
        key="num_concepts_widget", 
        help="Give an integer in the range [3, 50]",
    )
    
    if num_concepts is not None:
        # 1. Initialize or Resize Session State
        if 'num_concepts' not in st.session_state or st.session_state['num_concepts'] != num_concepts:
            old_names = st.session_state.get('concept_names', [])
            old_matrix = st.session_state.get('weight_matrix', pd.DataFrame())
            
            # Adjust list size 
            new_names = old_names[:num_concepts]
            while len(new_names) < num_concepts:
                new_names.append(f"Concept_{len(new_names)+1}")
            
            # Create new matrix filled with zeros
            new_matrix = pd.DataFrame(np.zeros((num_concepts, num_concepts)), columns=new_names, index=new_names)
            
            # Copy old data where names overlap
            overlap = [name for name in old_names if name in new_names]
            if not old_matrix.empty and overlap:
                new_matrix.loc[overlap, overlap] = old_matrix.loc[overlap, overlap]
            
            # Update session state
            st.session_state['num_concepts'] = num_concepts
            st.session_state['concept_names'] = new_names
            st.session_state['weight_matrix'] = new_matrix
            
            # ADDED RERUN: Force sync immediately after resizing so the widget doesn't bug out!
            st.rerun() 

        st.subheader("Define concepts", divider="green")
        
        # 2. Concept Renaming Editor
        concepts_df = pd.DataFrame([st.session_state['concept_names']], columns=[f"C{i+1}" for i in range(num_concepts)])
        edited_concepts = st.data_editor(concepts_df, hide_index=True, key="concept_editor")
        new_concept_names = edited_concepts.iloc[0].tolist()
        
        # 3. Handle concept renaming safely using st.rerun()
        if new_concept_names != st.session_state['concept_names']:
            st.session_state['concept_names'] = new_concept_names
            st.session_state['weight_matrix'].columns = new_concept_names
            st.session_state['weight_matrix'].index = new_concept_names
            st.rerun() 
        
        st.subheader("Define weighted interconnections", divider="green")
        
        # 4. Format matrix for display 
        display_matrix = st.session_state['weight_matrix'].copy()
        display_matrix.insert(0, "-", display_matrix.index)
        
        # 5. Weight Matrix Editor
        edited_matrix_display = st.data_editor(
            display_matrix.style.apply(highlight_diagonal, axis=None),
            hide_index=True,
            disabled=["-"],
            column_config=fix_configs(display_matrix),
            key="matrix_editor" 
        )
        
        # 6. Save the finalized edits back to session state safely
        edited_matrix_display.set_index("-", inplace=True)
        edited_matrix = edited_matrix_display.astype(float)
        
        if not edited_matrix.equals(st.session_state['weight_matrix']):
            st.session_state['weight_matrix'] = edited_matrix
            st.rerun() 
        
        return st.session_state['weight_matrix'], True
    else:
        return None, False