import streamlit as st
import io
from PIL import Image


###callbacks, functions and static widgets
def _learning_task_on_change():
    """
    This is a callback method that changes the training state to avoid crushing the results, when a train has already implemented
    """
    st.session_state.training_finished = False


help_task = "Use the **classification** option if your FCM will categorize your data into distinct classes. \
    \nUse the **Timeseries forecasting** option if your dataset aims to forecast a value based on historical measurements\
    \nUse the **Regression** option if your FCM will try to predict a continuous variable.\
    \n\nThe difference between **Timeseries forecasting** and **Regression** is in the **formatting of the input dataset**. \
    Timeseries forecasting assumes that previous measumenents affect future values. \
        Thus, it utilizes historical timesteps as concepts to forecast future timesteps and reshapes the dataset so that the input $f(y^{t1}) = x^{t0} + y^{t0}$"


def sidebar_widgets_task():
    if st.session_state.uploaded:
        st.write("")
        st.sidebar.radio(
            "Select task",
            ["Classification", "Regression"],
            key="learning_task",
            help=help_task,
            on_change=_learning_task_on_change,
        )

        st.sidebar.info(f"You selected {st.session_state.learning_task}")
        st.sidebar.write("")

    else:
        st.sidebar.info("Import data to continue", icon="ℹ️")
    sidebar_help_learning()
    # sidebar_logo()


def sidebar_widgets_show_df():
    if st.session_state.uploaded:
        check_dataset = st.sidebar.toggle(
            "Show dataset", help="Show the working dataset"
        )
        if check_dataset:
            t1, t2, t3 = st.tabs(
                ["📃 Dataset", "📊 Dataset statistics", "🔍 Generic info"]
            )
            with t1:
                st.caption("Working dataset")
                st.dataframe(st.session_state.working_df)
            with t2:
                st.write(st.session_state.working_df.describe())
            with t3:
                buffer = io.StringIO()
                st.session_state.working_df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)


def sidebar_logo():
    # Footer
    logo = Image.open("imgs\\FuzzyMind logo -cropped.png")
    st.sidebar.image(logo, use_column_width=True)


def sidebar_help_home():
    st.sidebar.markdown(
        """
    ## 🏗️ How to Use
    1. Select a tool from the sidebar above.
    2. Follow the in-page guides to use the tool.
    
    ---
    Need more help? [📖 Read the Manual](https://github.com/theotziol/fcm-app/blob/master/Manual.docx)
    """
    )


def sidebar_help_learning():
    st.sidebar.markdown(
        """
    ## 🏗️ How to use
    1. Upload a .csv dataset.
    2. Preprocess your data.
    3. Apply Neural-FCM learning. 
    ---
    Use the help (?) icon to understand widgets' functionality

    Need more help? [📖 Read the Manual](https://github.com/theotziol/fcm-app/blob/master/Manual.docx)
    """
    )


def sidebar_help_design_manually():
    st.sidebar.markdown(
        """
    ## 🏗️ How to use
    1. Define FCM concepts.
    2. Define weighted interconnections.
    3. See and modify the graph.
    4. Select inference parameters.
    5. Initialize concept values.
    6. Perform inference for what-if scenarios.
    ---
    Use the help (?) icon to understand widgets' functionality

    Need more help? [📖 Read the Manual](https://github.com/theotziol/fcm-app/blob/master/Manual.docx)
    """
    )


def sidebar_help_design_linguistic():
    st.sidebar.markdown(
        """
    ## 🏗️ How to use
    1. Define FCM concepts.
    2. Define causual interconnections.
    3. See and modify the graph.
    4. Defuzzify weights.
    5. Select inference parameters.
    6. Initialize concept values.
    7. Perform inference for what-if scenarios.
    ---
    Use the help (?) icon to understand widgets' functionality

    Need more help? [📖 Read the Manual](https://github.com/theotziol/fcm-app/blob/master/Manual.docx)
    """
    )


def sidebar_help_design_with_upload():
    st.sidebar.markdown(
        """
    ## 🏗️ How to use
    1. Upload an FCM as .csv.
    2. See and modify the graph.
    3. Select inference parameters.
    4. Initialize concept values.
    5. Perform inference for what-if scenarios.
    ---
    Use the help (?) icon to understand widgets' functionality

    Need more help? [📖 Read the Manual](https://github.com/theotziol/fcm-app/blob/master/Manual.docx)
    """
    )


def sidebar_help_design_with_upload_linguistic():
    st.sidebar.markdown(
        """
    ## 🏗️ How to use
    1. Upload an FCM as .csv and MFs in as .json.
    2. Defuzzify causual interconnections.
    3. See and modify the graph.
    4. Select inference parameters.
    5. Initialize concept values.
    6. Perform inference for what-if scenarios.
    ---
    Use the help (?) icon to understand widgets' functionality

    Need more help? [📖 Read the Manual](https://github.com/theotziol/fcm-app/blob/master/Manual.docx)
    """
    )


def sidebar_help_knowledge_aggregation():
    st.sidebar.markdown(
        """
    ## 🏗️ How to use
    1. Upload multiple FCMs, MFs pairs as .csv and .json.
    2. Check the aggregated causual interconnections
    2. Defuzzify causual interconnections.
    3. See and modify the graph.
    4. Select inference parameters.
    5. Initialize concept values.
    6. Perform inference for what-if scenarios.
    ---
    Use the help (?) icon to understand widgets' functionality

    Need more help? [📖 Read the Manual](https://github.com/theotziol/fcm-app/blob/master/Manual.docx)
    """
    )
