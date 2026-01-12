import streamlit as st
import pandas as pd
from collections import OrderedDict

st.set_page_config(page_title="HP EasyAnalyze", layout="wide", page_icon="🧊")
st.title("HP EasyAnalyze 🧊")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    
    if not pd.api.types.is_datetime64_any_dtype(df["recordeddate"]):
        df["recordeddate"] = pd.to_datetime(df["recordeddate"], errors="coerce")
    
    min_date = df["recordeddate"].min()
    max_date = df["recordeddate"].max()

    st.sidebar.write("Please select date ranges 🗓️")
    date_range1 = st.sidebar.date_input(
        "Date Range 1",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    date_range2 = st.sidebar.date_input(
        "Date Range 2",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    restricted_cols = [
        "recordeddate",
        "Sum of SurveyCount",
        "Sum of SurveyCount2",
        "Sum of TCR_Yes",
        "Sum of TCR_No",
        "Sum of CSAT_Num",
    ]
    
    available_dims = [col for col in df.columns if col not in restricted_cols]

    if "active_filters" not in st.session_state:
        st.session_state.active_filters = OrderedDict()

    st.markdown("### Available Dimensions 🔍")
    cols = st.columns(3)
    for i, col in enumerate(available_dims):
        if cols[i % 3].button(col, key=f"addbtn_{col}"):
            if col not in st.session_state.active_filters:
                st.session_state.active_filters[col] = None
                st.rerun()

    if st.session_state.active_filters:
        st.sidebar.markdown("### Active Filters 🔍")
        for dim in list(st.session_state.active_filters.keys()):
            mask = pd.Series(True, index=df.index)
            for other_dim, other_val in st.session_state.active_filters.items():
                if other_val is not None:
                    mask &= df[other_dim].astype(str).eq(str(other_val))
            temp_df = df.loc[mask]
            possible_vals = sorted(
                [str(x) for x in temp_df[dim].dropna().unique()],
                key=lambda x: x.lower()
            )

            selected_val = st.sidebar.selectbox(
                f"{dim} Filter",
                options=["(All)"] + possible_vals,
                key=f"sel_{dim}",
            )

            st.session_state.active_filters[dim] = None if selected_val == "(All)" else selected_val

            reset_col, remove_col = st.sidebar.columns(2)
            if reset_col.button(f"🔁 Reset {dim}", key=f"reset_{dim}"):
                st.session_state.active_filters[dim] = None
                st.rerun()
            if remove_col.button(f"❌ Remove {dim}", key=f"remove_{dim}"):
                del st.session_state.active_filters[dim]
                st.rerun()
    else:
        st.sidebar.info("No dimensions selected.")

    filtered_df = df.copy()
    for dim, val in st.session_state.active_filters.items():
        if val is not None:
            filtered_df = filtered_df[filtered_df[dim].astype(str) == str(val)]

    def calc_group_stats(dataframe, start_date, end_date, group_cols):
        mask = (dataframe["recordeddate"] >= pd.to_datetime(start_date)) & \
               (dataframe["recordeddate"] <= pd.to_datetime(end_date))
        filtered = dataframe.loc[mask].copy()

        for col in ["Sum of SurveyCount", "Sum of TCR_Yes", "Sum of CSAT_Num"]:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(0)

        total_survey_count = filtered["Sum of SurveyCount"].sum()

        if group_cols:
            grouped = filtered.groupby(group_cols).agg({
                "Sum of TCR_Yes": "sum",
                "Sum of CSAT_Num": "sum",
                "Sum of SurveyCount": "sum",
            }).reset_index()
        else:
            grouped = filtered.agg({
                "Sum of TCR_Yes": "sum",
                "Sum of CSAT_Num": "sum",
                "Sum of SurveyCount": "sum",
            }).to_frame().T
            grouped.insert(0, "All Data", "All Data")
            group_cols = ["All Data"]

        grouped["Sum of SurveyCount2"] = (
            grouped["Sum of SurveyCount"] / total_survey_count * 100
            if total_survey_count else 0.0
        )

        grouped["TCR%"] = grouped.apply(
            lambda r: (r["Sum of TCR_Yes"] * 100 / r["Sum of SurveyCount"])
            if r["Sum of SurveyCount"] else 0.0,
            axis=1,
        )

        grouped["CSAT%"] = grouped.apply(
            lambda r: (r["Sum of CSAT_Num"] * 100 / r["Sum of SurveyCount"])
            if r["Sum of SurveyCount"] else 0.0,
            axis=1,
        )

        grouped["Weightage (Sumproduct)"] = (grouped["Sum of SurveyCount2"] * grouped["TCR%"]) / 100

        total_row = pd.Series({
            group_cols[0]: "Grand Total",
            "Sum of SurveyCount": grouped["Sum of SurveyCount"].sum(),
            "Sum of SurveyCount2": grouped["Sum of SurveyCount2"].sum(),
            "TCR%": (
                grouped["Sum of TCR_Yes"].sum() * 100 / grouped["Sum of SurveyCount"].sum()
                if grouped["Sum of SurveyCount"].sum() else 0.0
            ),
            "CSAT%": (
                grouped["Sum of CSAT_Num"].sum() * 100 / grouped["Sum of SurveyCount"].sum()
                if grouped["Sum of SurveyCount"].sum() else 0.0
            ),
        })

        grouped = grouped[group_cols + [
            "Sum of SurveyCount",
            "Sum of SurveyCount2",
            "TCR%",
            "CSAT%",
            "Weightage (Sumproduct)"
        ]]

        return pd.concat([grouped, total_row.to_frame().T], ignore_index=True)

    active_keys = list(st.session_state.active_filters.keys())
    group_cols = active_keys if active_keys else []

    stats1 = calc_group_stats(filtered_df, date_range1[0], date_range1[1], group_cols)
    stats2 = calc_group_stats(filtered_df, date_range2[0], date_range2[1], group_cols)

    base1 = stats1.iloc[:-1]
    base2 = stats2.iloc[:-1]

    merged = base1.merge(base2, on=group_cols, suffixes=(" R1", " R2")) if group_cols else base1

    for m in ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]:
        merged[f"{m} Diff"] = merged[f"{m} R2"] - merged[f"{m} R1"]

    metrics_with_subcols = [
        "Sum of SurveyCount",
        "Sum of SurveyCount2",
        "TCR%",
        "CSAT%",
        "Weightage (Sumproduct)",
    ]

    data_dict = {}
    for col in group_cols:
        data_dict[(col, "")] = merged[col]

    for m in metrics_with_subcols:
        data_dict[(m, "R1")] = merged[f"{m} R1"]
        data_dict[(m, "R2")] = merged[f"{m} R2"]
        data_dict[(m, "Diff")] = merged[f"{m} Diff"]

    multi_df = pd.DataFrame(data_dict)
    multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns)

    def smart_format(val, col):
        try:
            val = float(val)
            if "%" in col[0]:
                return f"{val:.2f}%"
            return f"{val:.2f}"
        except:
            return val

    styled_multi_df = multi_df.style.format({
        col: (lambda v, c=col: smart_format(v, c))
        for col in multi_df.columns
    })

    st.subheader("Comparison Table 📚")
    st.dataframe(styled_multi_df, use_container_width=True)

else:
    st.info("Upload an Excel file to get started.")
