import streamlit as st
import sys
from copy import deepcopy as dc

sys.path.insert(1, "../fcm_codes")
from fcm_codes.general_functions import *
import skfuzzy
import json


dic_variables_caption = {
    5: "(:red[-High], :red[-Low],  None,  :blue[+Low], :blue[+High])",
    7: "(:red[-High], :red[-Medium], :red[-Low],  None,  :blue[+Low], :blue[+Medium], :blue[+High])",
    11: "(:red[-Very High], :red[-High], :red[-Medium], :red[-Low], :red[-Very Low],  None,  :blue[+Very Low], :blue[+Low], :blue[+Medium], :blue[+High], :blue[+Very High])",
}

dic_variables = {
    5: ["-High", "-Low", "None", "+Low", "+High"],
    7: ["-High", "-Medium", "-Low", "None", "+Low", "+Medium", "+High"],
    11: [
        "-Very High",
        "-High",
        "-Medium",
        "-Low",
        "-Very Low",
        "None",
        "+Very Low",
        "+Low",
        "+Medium",
        "+High",
        "+Very High",
    ],
}

dic_final = {
    "method": None,
    "range": [-1.0, 1.0],
    "step": 0.01,
    "memberships": {},
}

dic_defuz = {
    "Centroid": "centroid",
    "Bisector": "bisector",
    "Mean of Maximum (MoM)": "mom",
}


## Design manually tab main function
def fuzzy_sets():
    """
    The streamlit widgets for creating linguistic terms
    It utilizes two functions from within this module to initialize and modify the mf's parameters.
    It returns a dictionairy that contains all the necessary information for rebuilding the linguistic sets
    """
    st.subheader("Define fuzzy causality", divider="green")
    # do popup containers
    num_fuzzy_variables = st.radio(
        "Select the linguistic terms",
        [5, 7, 11],
        captions=[
            dic_variables_caption[i].replace(" ", "\n")
            for i in dic_variables_caption.keys()
        ],
        horizontal=True,
    )
    with st.expander("Configure membership function parameters..."):
        membership = st.selectbox(
            "Select the type of membership Function",
            ["Triangular", "Trapezoidal", "Gaussian"],
            index=0,
        )
        st.write("$\\mathbb{U} = [-1, 1]$")
        dic = initialize_fuzzy_memberships(
            dic_variables[num_fuzzy_variables], membership
        )
        final_dic = modify_fuzzy_memberships(dic)
    return final_dic


@st.cache_data
def initialize_fuzzy_memberships(memberships, method):
    """
    This function initializes the parameters of the fuzzy memberhsips.
    It divides the universe of discource (U) into evenly spaced chunks to position the highest membership values.
    The starting and ending points of the membership functions are then automatically constructed
    Args:
        memberships: a list of of linguistic terms (e.g., ['+High', '-Low']).
        method: string, the type of the membership function. Currently accepts ['Triangular', 'Trapezodial', 'Gaussian']
    Returns:
        dic
    """
    new_dic = dc(dic_final)
    U = new_dic["range"]
    new_dic["method"] = method
    mids = np.linspace(U[0], U[-1], len(memberships))
    mids = np.round(mids, 2)
    starting_points = np.insert(mids, 0, mids[0])
    starting_points = np.delete(starting_points, -1)

    finishing_points = np.insert(mids, -1, mids[-1])
    finishing_points = np.delete(finishing_points, 0)

    lower_trapezoidal = []
    upper_trapezoidal = []
    for i, value in enumerate(mids):
        # try except to avoid zero division
        try:
            value_low = (mids[i] + starting_points[i]) / 2
        except:
            value_low = starting_points[i]

        try:
            value_high = (mids[i] + finishing_points[i]) / 2
        except:
            value_high = finishing_points[i]

        lower_trapezoidal.append(value_low)
        upper_trapezoidal.append(value_high)

    sigma = (2 / len(memberships)) / 2

    if method == "Triangular":
        for i, mf in enumerate(memberships):
            new_dic["memberships"][mf] = [
                starting_points[i],
                mids[i],
                finishing_points[i],
            ]
    elif method == "Trapezoidal":
        for i, mf in enumerate(memberships):
            new_dic["memberships"][mf] = [
                starting_points[i],
                lower_trapezoidal[i],
                upper_trapezoidal[i],
                finishing_points[i],
            ]
    elif method == "Gaussian":
        for i, mf in enumerate(memberships):
            new_dic["memberships"][mf] = [mids[i], sigma]
    return new_dic


def modify_fuzzy_memberships(initialized_dic):
    """
    This function contains widgets to modify the fuzzy memberships
    Returns:
        a dictionairy that contains all the necessary information for rebuilding the fuzzy set, with keys = ['method', 'range', 'step', 'memberships']
    """
    col1, col2 = st.columns(2)
    method = initialized_dic["method"]
    array = np.linspace(
        initialized_dic["range"][0],
        initialized_dic["range"][-1],
        int(
            (initialized_dic["range"][-1] - initialized_dic["range"][0])
            / initialized_dic["step"]
        ),
    )

    with col1:
        new_dic = dc(initialized_dic)
        mfs = {}
        for i in new_dic["memberships"].keys():
            if method == "Triangular":
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    # st.write('Define **start** and **ending**:')
                    # here the user changes the start and the end of the triangle
                    start, end = st.slider(
                        f"**{i}** mf",
                        new_dic["range"][0],
                        new_dic["range"][1],
                        (new_dic["memberships"][i][0], new_dic["memberships"][i][-1]),
                    )
                    new_dic["memberships"][i][0] = start
                    new_dic["memberships"][i][-1] = end
                with col1_2:
                    # here the user changes the mid of the triangle
                    # st.write('Define **middle**:')
                    mid = st.slider(
                        f"**{i}** mf", start, end, new_dic["memberships"][i][1]
                    )
                    new_dic["memberships"][i][1] = mid
                    mfs[i] = skfuzzy.trimf(array, new_dic["memberships"][i])

            elif method == "Trapezoidal":
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    # st.write('Define **start** and **ending**:')
                    start, end = st.slider(
                        f"**{i}** mf",
                        new_dic["range"][0],
                        new_dic["range"][1],
                        (new_dic["memberships"][i][0], new_dic["memberships"][i][-1]),
                    )
                    new_dic["memberships"][i][0] = start
                    new_dic["memberships"][i][-1] = end
                with col1_2:
                    # st.write('Define **start** and **ending** of the trapezoids:')
                    start_center, end_center = st.slider(
                        f"**{i}** mf",
                        start,
                        end,
                        (new_dic["memberships"][i][1], new_dic["memberships"][i][2]),
                    )
                    new_dic["memberships"][i][1] = start_center
                    new_dic["memberships"][i][2] = end_center
                    mfs[i] = skfuzzy.trapmf(array, new_dic["memberships"][i])

            elif method == "Gaussian":
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    # st.write('Define **mean**:')
                    mean = st.slider(
                        f"**{i}** mf",
                        new_dic["range"][0],
                        new_dic["range"][1],
                        new_dic["memberships"][i][0],
                    )
                    new_dic["memberships"][i][0] = mean
                with col1_2:
                    # st.write('Define $\\sigma$:')
                    sigma = st.slider(
                        f"**{i}** mf",
                        new_dic["range"][0],
                        new_dic["range"][1],
                        new_dic["memberships"][i][1],
                    )
                    new_dic["memberships"][i][-1] = sigma
                    mfs[i] = skfuzzy.gaussmf(
                        array,
                        new_dic["memberships"][i][0],
                        new_dic["memberships"][i][-1],
                    )

    df = pd.DataFrame(mfs, index=array)
    with col2:
        for i in range(len(mfs.keys()) // 2):
            st.write("")
        st.caption("Fuzzy Membership Functions Plot")
        st.line_chart(df)
        json_data = json.dumps(new_dic)
        st.json(json_data, expanded=False)
        st.download_button(
            "Download JSON",
            json_data,
            "fuzzy_info.json",
            mime="application/json",
        )
    return new_dic


def manual_tab_linguistic(dic):
    """
    The main tab for manual linguistic fcm construction
    """
    st.subheader("Define the total number of concepts", divider="blue")
    num_concepts = st.number_input(
        "Give the number of concepts",
        min_value=3,
        max_value=50,
        value=None,
        help="Give an integer in the range [3, 50]",
    )
    if num_concepts != None:
        st.subheader("Define concepts", divider="blue")
        columns_df = create_weight_matrix_columns(num_concepts)
        edited_columns = st.data_editor(columns_df, hide_index=True)
        st.subheader("Define linguistic interconnections", divider="green")
        mfs = list(dic["memberships"].keys())
        weight_matrix_df = create_linguistic_weight_matrix(
            num_concepts, edited_columns.values.tolist(), mfs
        )
        edited_matrix = st.data_editor(
            weight_matrix_df.style.apply(highlight_diagonal, axis=None),
            hide_index=True,
            disabled=["-"],
            column_config=fix_configs_linguistic(weight_matrix_df, mfs),
        )
        edited_matrix.set_index("-", inplace=True)
        return edited_matrix, True

    else:
        return None, False


def defuzzification_single(edited_matrix, final_dic):
    """
    Defuzzifies the weight matrix.
    """
    st.subheader("Defuzzification of weight matrix", divider="blue")
    defuzification_method = st.selectbox(
        "Select the defuzzification method",
        ["Centroid", "Bisector", "Mean of Maximum (MoM)"],
    )

    new_matrix = np.zeros(edited_matrix.shape)
    columns = edited_matrix.columns

    mfs = _create_mfs_dic(final_dic)

    for i, col in enumerate(edited_matrix.columns):
        for ii, col2 in enumerate(edited_matrix.columns):
            new_matrix[i, ii] = skfuzzy.defuzz(
                mfs["array"],
                mfs[edited_matrix.iloc[i, ii]],
                dic_defuz[defuzification_method],
            )
    df_matrix = pd.DataFrame(new_matrix, columns=columns, index=columns).round(2)
    return df_matrix


@st.cache_data
def _create_mfs_dic(final_dic):
    """
    function to create a dictionairy that contains the mfs arrays.
    Returns:
        __dict__ : keys = [mf1, mf2, ..., mfn], values = [array1, array2]
    """
    method = final_dic["method"]
    array = np.linspace(
        final_dic["range"][0],
        final_dic["range"][-1],
        int((final_dic["range"][-1] - final_dic["range"][0]) / final_dic["step"]),
    )
    mfs = {}

    for mf in final_dic["memberships"].keys():
        if method == "Triangular":
            mfs[mf] = skfuzzy.trimf(array, final_dic["memberships"][mf])
        elif method == "Trapezoidal":
            mfs[mf] = skfuzzy.trapmf(array, final_dic["memberships"][mf])
        elif method == "Gaussian":
            mfs[mf] = skfuzzy.gaussmf(
                array, final_dic["memberships"][mf][0], final_dic["memberships"][mf][-1]
            )
    mfs["array"] = array
    return mfs


### Knowledge aggregation function


def aggregation_info_display(dic_uploads):
    """
    This component displays the aggregation info of the uploaded matrices
    """
    with st.expander("Aggregation info..."):
        total_concepts, combined_df, succesfull_pairs, mfs_list = aggregation_info(
            dic_uploads
        )
        st.caption("Total defined concepts")
        st.dataframe(total_concepts, hide_index=True)
        st.caption(
            f"**The aggregation matrix**. Each cell contains the linguistic variables that were given per expert/stakeholder and found in {succesfull_pairs}. \
            **Undefined** indicates that no variable was passed in the corresponding file for this connection."
        )
        st.dataframe(combined_df)
        for i, mf_dic in enumerate(mfs_list):

            to_plot = dc(
                mf_dic
            )  # create a deep copy to pop the array key that has no plotting meaning
            index = to_plot.pop("array")
            df = pd.DataFrame(to_plot, index=index)
            st.caption(
                f"Fuzzy MFs based on the expert's opinion found in '{succesfull_pairs[i]}' files"
            )
            st.line_chart(df)
        # st.line_chart(np.fmax(df['-High'], df['-Medium']))
        dummy_df, stored_mfs = aggregate(combined_df, mfs_list)
        st.caption("Check the aggregated membership functions")
        col1, col2 = st.columns(2)
        with col1:
            concept1 = st.selectbox(
                "Select the first concept", dummy_df.columns, index=None
            )
        with col2:
            concept2 = st.selectbox(
                "Select the second concept", dummy_df.columns, index=None
            )
        if concept1 != None and concept2 != None:
            index = dummy_df.loc[concept1][concept2]
            st.caption("Suggested by experts membership functions")
            ignore = ["array", "aggregated"]
            dfs = []
            for key in stored_mfs[index].keys():
                if key not in ignore:
                    df = pd.DataFrame(
                        stored_mfs[index][key], index=stored_mfs[index]["array"]
                    )
                    dfs.append(df)
            for df in dfs:
                st.line_chart(df)

            st.caption("Aggregated membership function")
            df_aggr = pd.DataFrame(
                stored_mfs[index]["aggregated"], index=stored_mfs[index]["array"]
            )
            st.line_chart(df_aggr, color=(220, 0, 10))
    return dummy_df, stored_mfs


def defuzification_widgets(dummy_df, stored_mfs):
    """
    Provides the widgets for the defuzification. Returns the defuzified weight matrix.
    """
    st.subheader("Defuzzification of the weight matrix", divider="blue")
    defuzification_method = st.selectbox(
        "Select the defuzzification method",
        ["Centroid", "Bisector", "Mean of Maximum (MoM)"],
    )
    defuzified_matrix = defuzzify_aggregated(
        dummy_df, stored_mfs, defuzification_method
    )
    st.caption("Defuzzified matrix")
    st.dataframe(defuzified_matrix)
    return defuzified_matrix


@st.cache_data
def defuzzify_aggregated(dummy_df, stored_mfs, defuzification_method):
    columns = dummy_df.columns
    defuzified_matrix = pd.DataFrame(
        np.zeros(dummy_df.shape), columns=columns, index=columns
    )
    for row in columns:
        for col in columns:
            index = dummy_df.loc[row][col]
            defuzified_matrix.loc[row][col] = skfuzzy.defuzz(
                stored_mfs[index]["array"],
                stored_mfs[index]["aggregated"],
                dic_defuz[defuzification_method],
            )
    defuzified_matrix = defuzified_matrix.round(2)
    return defuzified_matrix


@st.cache_data
def aggregation_info(dic_uploads):
    """
    This function aims to provide info regarding the aggregation by:

        1) returning a pd.DataFrame with the total concepts that were passed
        2) returning a single pd.DataFrame where each cell unifies the linguistic values that were given in the uploaded files.
        3) returning a list of the paired files.
        4) returning a list of mfs dics.

    """
    pairs = [
        i for i in dic_uploads.keys() if dic_uploads[i] != None
    ]  # None value is passed when an uploaded file failed to be readen
    total_unique_concepts = []
    mfs_list = []
    for key in pairs:
        concepts = dic_uploads[key][0].columns
        mfs_info_dic = dic_uploads[key][1]
        mfs_list.append(_create_mfs_dic(mfs_info_dic))
        for i in concepts:
            if i not in total_unique_concepts:
                total_unique_concepts.append(i)

    df_columns = pd.DataFrame(
        [total_unique_concepts],
        columns=[f"C{i}" for i in range(len(total_unique_concepts))],
    )

    df = pd.DataFrame(
        np.empty((len(total_unique_concepts), len(total_unique_concepts))),
        index=total_unique_concepts,
        columns=total_unique_concepts,
    )
    combined_df = df.applymap(lambda x: [])

    for key in pairs:
        matrix = dic_uploads[key][0]
        # print(matrix)
        for con_row in total_unique_concepts:
            for con_column in total_unique_concepts:
                if con_row not in list(matrix.columns) or con_column not in list(
                    matrix.columns
                ):
                    combined_df.loc[con_row][con_column].append("Undefined")
                else:
                    combined_df.loc[con_row][con_column].append(
                        matrix.loc[con_row][con_column]
                    )

    return df_columns, combined_df, pairs, mfs_list


@st.cache_data
def aggregate(df_matrix, mfs, undefined="Undefined"):
    """
    This function aggregates the mfs. Initially, a new (dummy) dataframe is created that each cell value has an incremental integer which corresponds to a key in a dictionairy.
    The dictionairy stores the initial mfs and the aggregated mf for visualization purposes.
    """
    columns = df_matrix.columns
    dummy_array = [i + 1 for i in range(len(columns) * len(columns))]
    stored_mfs = {key + 1: {} for key in range(len(columns) * len(columns))}
    dummy_array = np.array(dummy_array).reshape((len(columns), len(columns)))
    dummy_df = pd.DataFrame(dummy_array, columns=columns, index=columns)
    print(stored_mfs)

    for row in columns:
        for column in columns:
            values = df_matrix.loc[row][column]
            index = dummy_df.loc[row][column]
            vals_keys = []

            for i, val in enumerate(values):
                if val == undefined:
                    continue
                else:
                    stored_mfs[index]["array"] = mfs[i]["array"]
                    stored_mfs[index][val] = mfs[i][val]
                    vals_keys.append(val)

            aggregated = stored_mfs[index][vals_keys[0]]
            if len(vals_keys) > 1:
                for i in range(len(vals_keys) - 1):
                    aggregated = np.fmax(
                        aggregated, stored_mfs[index][vals_keys[i + 1]]
                    )
            stored_mfs[index]["aggregated"] = aggregated
    return dummy_df, stored_mfs
