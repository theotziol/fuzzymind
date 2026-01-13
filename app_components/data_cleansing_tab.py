import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import io


def datacleansing_widgets():
    st.subheader("Data Cleansing", divider="blue")
    nan_values()
    outlier_removal()
    manual_proccess = st.toggle(
        "Manual processing",
        help="Provides an editable table for proccesing tables cells manually",
    )
    if manual_proccess:
        st.subheader("Manual processing", divider="gray")
        st.session_state.working_df = st.data_editor(
            st.session_state.working_df, on_change=data_editor_callback
        )


def nan_values():
    columns_na = [
        i
        for i in st.session_state.working_df.columns
        if st.session_state.working_df[i].isna().any()
    ]
    df = st.session_state.working_df.copy()
    imputation_methods = {
        "Interpolation": "Fill NaN values using an interpolation method",
        "Ffil": "Fill NA/NaN values by propagating the last valid observation to next valid",
        "Bfil": "Fill NA/NaN values by using the next valid observation to fill the gap.",
        "Value": "Fill NA/NaN values with a specific value.",
        "Statistics": "Fill NA/NaN values by using a statistic based value.",
    }

    if len(columns_na) > 0:
        st.info("Delete rows/columns, or impute values to columns with missing data")
        with st.expander("Imputation..."):
            column = st.selectbox("Select **column** to process", columns_na, None)
            if column is not None:
                st.write(f"You selected the {column} column")
            method = st.selectbox(
                "Select the **imputation method**",
                list(imputation_methods.keys()),
                None,
                key="fill_method",
            )
            if method is not None and column is not None:
                fig, axs = plt.subplots(figsize=(12, 4))
                st.write(imputation_methods[method])
                if method == "Value":
                    value, numeric = value_imputation_widgets()
                    df["imputed_values"] = df[column].fillna(value)
                    text_dtypes = ("object", "bool")
                    if numeric and df[column].dtype.name not in text_dtypes:
                        df.plot(y=["imputed_values", column], ax=axs)
                        st.pyplot(fig, dpi=500)
                    submit = st.button(
                        "Submit",
                        key="submit_val",
                        on_click=submit_imputation_value,
                        args=(
                            column,
                            value,
                        ),
                    )
                    if submit:
                        st.success(
                            f'NaN cells were succesfully imputed in {column} with the value "{value}."'
                        )

                elif method == "Statistics":
                    value = statistics_imputation_widgets(column)
                    df["imputed_values"] = df[column].fillna(value)
                    df.plot(y=["imputed_values", column], ax=axs)
                    st.pyplot(fig, dpi=500)
                    submit = st.button(
                        "Submit",
                        key="submit_stats",
                        on_click=submit_imputation_value,
                        args=(column, value),
                    )
                    if submit:
                        st.success(
                            f'NaN cells were succesfully imputed in {column} with the value "{value}."'
                        )

                elif method == "Interpolation":
                    interp_method = interpolation_imputation_widgets()
                    df["imputed_values"] = df[column].interpolate(interp_method)
                    df.plot(y=["imputed_values", column], ax=axs)
                    st.pyplot(fig, dpi=500)
                    submit = st.button(
                        "Submit",
                        key="submit_inter",
                        on_click=submit_interpolation_value,
                        args=(column, interp_method),
                    )
                    if submit:
                        st.success(
                            f'NaN cells were succesfully imputed in {column} with the "{interp_method}" interpolation method.'
                        )

                elif method == "Ffil":
                    df["imputed_values"] = df[column].ffill()
                    df.plot(y=["imputed_values", column], ax=axs)
                    st.pyplot(fig, dpi=500)
                    submit = st.button(
                        "Submit",
                        key="submit_ffil",
                        on_click=submit_ffil,
                        args=(column,),
                    )
                    if submit:
                        st.success(
                            f"NaN cells were succesfully imputed in {column} by propagating forward the previous values."
                        )
                elif method == "Bfil":
                    df["imputed_values"] = df[column].bfill()
                    df.plot(y=["imputed_values", column], ax=axs)
                    st.pyplot(fig, dpi=500)
                    submit = st.button(
                        "Submit",
                        key="submit_bfil",
                        on_click=submit_bfil,
                        args=(column,),
                    )
                    if submit:
                        st.success(
                            f"NaN cells were succesfully imputed in {column} by propagating backward the next values."
                        )

        with st.expander("Delete rows/columns"):
            col1, col2 = st.columns(2)
            with col1:
                delete_column = st.selectbox(
                    "Delete a column with NaN values", columns_na, None
                )
                if delete_column is not None:
                    st.write(f"You selected the {delete_column} to be deleted")
                    delete_col_button = st.button(
                        "Submit",
                        key="del_col",
                        on_click=submit_deletion_col,
                        args=(delete_column,),
                    )
            with col2:
                st.write("Dischard **all rows** with missing values.")
                delete_all_rows_button = st.button(
                    "Dischard rows", key="del_rows", on_click=submit_deletion_rows
                )

    else:
        st.info("No NaN values were found in dataset")
    col1, col2, col3 = st.columns((0.3, 0.4, 0.3))
    with col2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)


def outlier_removal():
    st.subheader("Outlier removal", divider="gray")
    with st.expander("Parameters...", expanded=False):
        st.info("Currently only statistic based removal is supported")

        numerical_cols = st.session_state.working_df.select_dtypes(
            include=np.number
        ).columns.tolist()
        # not_plotting = ("object", "bool")
        # columns = [
        #     col
        #     for col in st.session_state.working_df.columns
        #     if col not in not_plotting
        # ]
        columns = numerical_cols
        column = st.selectbox("Select column for filtering", columns, None)
        if column is not None:
            array = st.session_state.working_df[column].to_numpy()
            z = np.abs((array - array.mean()) / array.std())
            z_score = st.slider(
                "Select the Z-Score for filtering",
                min_value=1.5,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="This value/score indicates how far is the data point from the mean.\
                    After setting up a threshold value the z score values of data points are utilized to define the outliers. Zscore = (data_point -mean) / std. deviation",
            )
            outlier_indinces = np.where(z > z_score)[0]
            fig, axs = plt.subplots(figsize=(12, 4))
            axs.plot(st.session_state.working_df[column])
            axs.scatter(
                st.session_state.working_df.index[outlier_indinces],
                st.session_state.working_df[column].iloc[outlier_indinces],
                c="r",
            )
            st.pyplot(fig, dpi=500)
            tb1, tb2 = st.tabs(["Convert outliers", "Delete outliers"])
            with tb1:
                st.write("Select a replacing value for the identified outliers")
                outlier_processing_widgets(column, outlier_indinces)

            with tb2:
                st.write(
                    f"Discard **all rows in dataset** where outliers for column {column} were identified\n"
                )
                col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
                discard = st.button(
                    "Discard",
                    on_click=discard_outliers_callback,
                    args=(column, outlier_indinces),
                )
                if discard:
                    st.success(
                        f"{len(outlier_indinces)} rows have been succesfully discarded from dataset"
                    )


def value_imputation_widgets():
    """
    This function is used for providing widgets when the users decides to manually provide a value for imputation.
    Returns either a numeric or a text value
    """
    bool_num = st.toggle("Numeric value", True, help="Deactivate if text will be given")

    if bool_num:
        value = st.number_input("Give a numeric value", 0.0)
    else:
        value = st.text_input(
            "Provide a string type value",
        )
    return value, bool_num


def statistics_imputation_widgets(column):
    """
    This function is used for providing widgets when the users decides to manually provide a value for imputation.
    Returns either a numeric or a text value
    """

    try:
        mean = st.session_state.working_df[column].mean()
        median = st.session_state.working_df[column].median()
        maxx = st.session_state.working_df[column].max()
        minn = st.session_state.working_df[column].min()
        dic = {"Mean": mean, "Median": median, "Max": maxx, "Min": minn}
        st.text(dic)
        selection = st.selectbox(
            "Select statistic method for imputation",
            list(dic.keys()),
        )

        return dic[selection]
    except Exception as e:
        st.warning(e)
        return None


def interpolation_imputation_widgets():
    """
    This function is used for providing widgets when the users decides to manually provide a value for imputation.
    Returns either a numeric or a text value
    """

    methods = {
        "Linear": "linear",
        "Nearest": "nearest",
        # 'Cubic' : 'cubic',
        # 'Quadratic' : 'quadratic',
    }
    selection = st.selectbox("Select method for imputation", list(methods.keys()))
    return methods[selection]


def outlier_processing_widgets(column, indexes):
    replace_method = st.radio(
        "Provide a replacing method",
        ["Mean", "Interpolation", "Other"],
        index=None,
        horizontal=True,
        captions=[
            "Replace with the mean (average)",
            "Replace with values calculated with linear interpolation",
            "Provide a value within column's range",
        ],
    )
    if replace_method == "Other":
        minimum = np.round(st.session_state.working_df[column].min(), 4)
        maximum = np.round(st.session_state.working_df[column].max(), 4)
        median = np.round(st.session_state.working_df[column].median(), 4)
        total_range = np.abs(
            st.session_state.working_df[column].max()
            - st.session_state.working_df[column].min()
        )
        values = st.number_input(
            "Provide an input",
            minimum,
            maximum,
            median,
        )
        submit = st.button(
            "Submit", on_click=submit_outlier_values, args=(values, column, indexes)
        )
        if submit:
            st.success("Outliers have been succesfully converted")

    elif replace_method == "Mean":
        values = st.session_state.working_df[column].mean()
        st.write(f"Mean value = {values}")
        submit = st.button(
            "Submit", on_click=submit_outlier_values, args=(values, column, indexes)
        )
        if submit:
            st.success("Outliers have been succesfully converted")

    elif replace_method == "Interpolation":
        series_copy = st.session_state.working_df[column].copy()
        series_copy.iloc[indexes] = np.nan
        series_copy.interpolate(inplace=True)
        submit = st.button(
            "Submit",
            on_click=submit_outlier_values_interpolation,
            args=(column, indexes, series_copy),
        )
        if submit:
            st.success("Outliers have been succesfully converted")


### submit callbaks for on click events
def submit_imputation_value(column, value):
    st.session_state.working_df[column].fillna(value, inplace=True)
    st.session_state["fill_method"] = None
    st.session_state.changed = True


def submit_interpolation_value(column, method):
    st.session_state.working_df[column].interpolate(method, inplace=True)
    st.session_state["fill_method"] = None
    st.session_state.changed = True


def submit_ffil(column):
    st.session_state.working_df[column].ffill(inplace=True)
    st.session_state["fill_method"] = None
    st.session_state.changed = True


def submit_bfil(column):
    st.session_state.working_df[column].bfill(inplace=True)
    st.session_state["fill_method"] = None
    st.session_state.changed = True


def submit_deletion_rows():
    st.session_state.working_df.dropna(inplace=True)
    st.session_state.changed = True


def submit_deletion_col(column):
    st.session_state.working_df.drop([column], axis=1, inplace=True)
    st.session_state.changed = True


def data_editor_callback():
    st.session_state.changed = True


def submit_outlier_values(values, column, indexes):
    st.session_state.changed = True
    st.session_state.working_df[column].iloc[indexes] = values


def submit_outlier_values_interpolation(column, indexes, copied_column):
    st.session_state.changed = True
    st.session_state.working_df[column].iloc[indexes] = copied_column[indexes]


def discard_outliers_callback(column, indexes):
    st.session_state.changed = True
    st.session_state.working_df.drop(
        st.session_state.working_df.index[indexes], inplace=True
    )
