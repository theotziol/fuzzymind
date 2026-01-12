### This script contains the widgets for the weight_matrix results tab


import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

sys.path.insert(1, "../fcm_codes")
sys.path.insert(1, "../app_components")
from fcm_codes.graphs import *
from fcm_codes.preprocessing import reverse_min_max
from app_components.fcm_graph_component import *


def weight_matrix_results(fold=None):
    if st.session_state.learning_algorithm == "Neural-FCM":
        if fold == None:
            x_test = st.session_state.input_df.iloc[
                int(len(st.session_state.input_df) * st.session_state.split_ratio) :
            ]
            y_test = st.session_state.output_df.iloc[
                int(len(st.session_state.output_df) * st.session_state.split_ratio) :
            ]
            weight_matrix_widgets_neuralfcm(st.session_state.model, x_test, y_test)
        else:
            model = st.session_state.kfold_dic[fold]
            test_index = st.session_state.kfold_dic[fold].test_index
            x_test = st.session_state.input_df.iloc[test_index]
            y_test = st.session_state.output_df.iloc[test_index]

            weight_matrix_widgets_neuralfcm(model, x_test, y_test)
    elif st.session_state.learning_algorithm == "Particle Swarm Optimization":
        pass


def weight_matrix_widgets_neuralfcm(model, x_test, y_test):
    st.info("Neural-FCM predicts a weight matrix for each instance.", icon="ℹ️")
    col1, col2 = st.columns(2)
    with col1:
        index = st.slider(
            "Select an instance from the test dataset", 0, len(x_test) - 1, 0
        )
        st.caption(f"Input: Testing instance {index}")
        values_dictionairy = {"Normalized" : x_test.iloc[index].to_numpy(), "Original" : reverse_min_max(x_test, st.session_state.non_norm_working_df).iloc[index].to_numpy()}
        show_df = pd.DataFrame(values_dictionairy, index=x_test.columns)
        st.dataframe(show_df)
        ### to do change the st.write for regression-forecasting
        if st.session_state.learning_task == "Classification":
            st.write(
                f"**Actual Class**: {y_test.columns[np.argmax(y_test.iloc[index])]}"
            )
            st.write(
                f"**Predicted Class**: {y_test.columns[np.argmax(model.predictions[index])]}, {np.round(model.predictions[index], 2)}"
            )
        else:
            st.write(f"**Actual value (norm)**: {np.round(model.test_y[index], 3)}")
            st.write(
                f"**Predicted value (norm)**: {np.round(model.predictions[index], 3)}, "
            )
            st.write(f"**Actual value**: {model.real_array_test[index]}")
            st.write(
                f"**Predicted value**: {np.round(model.predictions_actual[index], 3)}, "
            )

    with col2:
        fig_size = st.slider("Change figure size", 2, 14, 5)
        size = st.slider("Change figure text size", 4, 30, 8)
        denoise_boolean = st.toggle(
        "Denoise matrix",
        help="It sets the diagonal and the class rows to zero (0) values",
        )
        fig, axs = plt.subplots(figsize=(fig_size, fig_size))
        if denoise_boolean:
            plot_matrix = denoise_matrix(
            model.predicted_matrices[index, :, :, 0], x_test.columns, y_test
            )
            im, cbar = heatmap(
                plot_matrix.to_numpy(),
                x_test.columns,
                x_test.columns,
                ax=axs,
                cmap="coolwarm",
            )
            texts = annotate_heatmap(im, size)
            st.pyplot(fig, dpi=700)
            st.caption("Denoised FCM weight matrix.")
        else:
            plot_matrix =  pd.DataFrame(
            np.round(model.predicted_matrices[index, :, :, 0], 2),
            columns=x_test.columns,
            index=x_test.columns,
        )
            im, cbar = heatmap(
                model.predicted_matrices[index, :, :, 0],
                x_test.columns,
                x_test.columns,
                ax=axs,
                cmap="coolwarm",
            )
            texts = annotate_heatmap(im, size)
            st.pyplot(fig, dpi=500)
            st.caption("FCM weight matrix.")
        
    st.caption("Denoised FCM weight matrix.")
    st.dataframe(plot_matrix)
            

    # st.dataframe(pd.DataFrame(model.predicted_matrices[index, :, :, 0], columns = x_test.columns,index = x_test.columns))# index = x_test.columns


    graph_boolean = st.toggle(
        "Generate FCM graph", help="Visualize as a graph the weight matrix."
    )
    if graph_boolean:
        graph(plot_matrix)

    average_boolean = st.toggle(
        "Generate FCM test average",
        help="Visualize the ranges of the predicted weight matrices.",
    )
    if average_boolean:
        fig_size_av = st.slider("Change figure size", 2, 14, 6, key = "average_slider_figsize")
        size_av = st.slider("Change figure text size", 4, 30, 8, key = "average_slider_textsize")
        if st.session_state.learning_task == "Classification":
            st.pyplot(
                calculate_and_plot_stats_of_matrices(
                    model.predicted_matrices, x_test.columns, y_test, "Statistic Values", figsize=(2*fig_size_av + 1, fig_size_av), size=size_av
                ),
                dpi=500,
            )
            
        else:
            st.pyplot(
                calculate_and_plot_stats_of_matrices(
                    model.predicted_matrices,
                    x_test.columns,
                    np.zeros((len(y_test), len(st.session_state.output_columns))),
                    "Statistic Values", figsize=(2*fig_size_av + 1, fig_size_av), size=size_av
                ),
                dpi=700,
            )
            
        show_tables_statistics = st.toggle("Show tables", help = "Matrices as tables", key="show_average_tables")
        if show_tables_statistics:
            if st.session_state.learning_task == "Classification":
                mean_df, std_df = calculate_and_plot_stats_of_matrices(
                    model.predicted_matrices, x_test.columns, y_test, "Statistic Values", figsize=(2*fig_size_av + 1, fig_size_av), size=size_av, return_tables=True
                )
            else:
                mean_df, std_df = calculate_and_plot_stats_of_matrices(
                    model.predicted_matrices,
                    x_test.columns,
                    np.zeros((len(y_test), len(st.session_state.output_columns))),
                    "Statistic Values", figsize=(2*fig_size_av + 1, fig_size_av), size=size_av, return_tables=True
                )
            col1_mean, col2_std = st.columns(2)
            with col1_mean:
                st.caption("Average values across testing samples")
                st.dataframe(mean_df)
            with col2_std:
                st.caption("standard deviation values across testing samples")
                st.dataframe(std_df)
            


@st.cache_data
def denoise_matrix(array, _columns, y_test):
    array = array.copy()
    array = np.round(array, 2)
    # array = np.fill_diagonal(array.copy(), 0.0)
    for i in range(array.shape[0]):
        array[i, i] = 0.0

    if st.session_state.learning_task == "Classification":
        for i in range(y_test.shape[-1]):
            array[-(i + 1), :] = 0.0
    else:
        for i in range(y_test.shape[-1]):
            array[-len(st.session_state.output_columns), :] = 0.0
    return pd.DataFrame(array, columns=_columns, index=_columns)
