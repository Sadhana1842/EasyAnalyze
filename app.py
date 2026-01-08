import streamlit as st
import pandas as pd
from collections import OrderedDict
from pandas import IndexSlice as i_slice

st.set_page_config(page_title="HP EasyAnalyze", layout="wide", page_icon="🧊")
st.title("HP EasyAnalyze 🧊")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # Ensure recordeddate is datetime
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
    
    # ---------------- FILTERING LOGIC START ----------------
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

    # Dimension buttons main area
    st.markdown("### Available Dimensions 🔍")
    cols = st.columns(3)
    for i, col in enumerate(available_dims):
        if cols[i % 3].button(col, key=f"addbtn_{col}"):
            if col not in st.session_state.active_filters:
                st.session_state.active_filters[col] = None
                st.rerun()

    # Active filters sidebar
    if st.session_state.active_filters:
        st.sidebar.markdown("### Active Filters 🔍")
        for dim in list(st.session_state.active_filters.keys()):
            mask = pd.Series(True, index=df.index)
            for other_dim, other_val in st.session_state.active_filters.items():
                if other_val is not None:
                    mask &= df[other_dim].astype(str).eq(str(other_val))
            temp_df = df.loc[mask]
            possible_vals_raw = temp_df[dim].dropna().unique().tolist()
            possible_vals = sorted([str(x) for x in possible_vals_raw], key=lambda x: x.lower())

            current_val = st.session_state.active_filters[dim]
            if current_val is not None and str(current_val) not in possible_vals:
                current_val = None

            selected_val = st.sidebar.selectbox(
                f"{dim} Filter",
                options=["(All)"] + possible_vals,
                index=0 if current_val is None else (possible_vals.index(str(current_val)) + 1),
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
        st.sidebar.info("No dimensions selected. Click a dimension above to add it as a filter.")

    # Apply active filters to dataframe
    filtered_df = df.copy()
    for dim, val in st.session_state.active_filters.items():
        if val is not None:
            filtered_df = filtered_df[filtered_df[dim].astype(str) == str(val)]

    # ---------------- FILTERING LOGIC END ----------------
    
    def calc_group_stats(dataframe, start_date, end_date, group_cols):
        mask = (dataframe["recordeddate"] >= pd.to_datetime(start_date)) &\
               (dataframe["recordeddate"] <= pd.to_datetime(end_date))
        filtered = dataframe.loc[mask].copy()

        for col in ["Sum of SurveyCount", "Sum of TCR_Yes", "Sum of CSAT_Num"]:
            if col in filtered.columns:
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

        if total_survey_count == 0:
            grouped["Sum of SurveyCount2"] = 0.0
        else:
            grouped["Sum of SurveyCount2"] = grouped["Sum of SurveyCount"] / total_survey_count * 100
        
        grouped["Sum of SurveyCount2"] = grouped["Sum of SurveyCount2"].round(2)

        grouped["TCR%"] = grouped.apply(
            lambda r: (r["Sum of TCR_Yes"]*100.0/r["Sum of SurveyCount"]) if r["Sum of SurveyCount"] else 0.0, axis=1)
        grouped["CSAT%"] = grouped.apply(
            lambda r: (r["Sum of CSAT_Num"]*100.0/r["Sum of SurveyCount"]) if r["Sum of SurveyCount"] else 0.0, axis=1)
        grouped["Weightage (Sumproduct)"] = ((grouped["Sum of SurveyCount2"]/100)*grouped["TCR%"]).round(4)

        total_row = pd.Series({
            group_cols[0]: "Grand Total",
            "Sum of SurveyCount": grouped["Sum of SurveyCount"].sum(),
            "Sum of SurveyCount2": grouped["Sum of SurveyCount2"].sum(),
            "TCR%": (grouped["Sum of TCR_Yes"].sum() / grouped["Sum of SurveyCount"].sum()) if grouped["Sum of SurveyCount"].sum() else 0.0,
            "CSAT%": (grouped["Sum of CSAT_Num"].sum() / grouped["Sum of SurveyCount"].sum()) if grouped["Sum of SurveyCount"].sum() else 0.0,
        })

        grouped = grouped[group_cols + ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]]
        grouped = pd.concat([grouped, total_row.to_frame().T], ignore_index=True)
        return grouped

    try:
        active_keys = list(st.session_state.active_filters.keys())
        if active_keys:
            group_cols = active_keys
        else:
            group_cols = []

        stats1 = calc_group_stats(filtered_df, date_range1[0], date_range1[1], group_cols)
        stats2 = calc_group_stats(filtered_df, date_range2[0], date_range2[1], group_cols)
    except Exception as e:
        st.error(f"Can't calculate tables: {e}")

    base1 = stats1.iloc[:-1].reset_index(drop=True)
    base2 = stats2.iloc[:-1].reset_index(drop=True)

    if group_cols:
        merged = base1.merge(base2, on=group_cols, how="inner", suffixes=(" R1", " R2"))
    else:
        merged = base1.copy()
        merged = merged.add_suffix(" R1")
        for col in base2.columns:
            merged[f"{col} R2"] = base2[col].values

    # Compute calculated columns for Impact %, Mix Shift Impact, Score Impact
    merged["Impact %"] = merged["Weightage (Sumproduct) R2"] - merged["Weightage (Sumproduct) R1"]
    merged["Mix Shift Impact"] = (merged["TCR% R1"] / 100) * merged["Sum of SurveyCount2 R2"]
    merged["Score Impact"] = (merged["Sum of SurveyCount2 R1"] / 100) * merged["TCR% R2"]

    # Per-metric Diff columns (R2 - R1) for metrics that have R1/R2
    for m in ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]:
        merged[f"{m} Diff"] = merged[f"{m} R2"] - merged[f"{m} R1"]
        
    #To sort by R2 sample by default (Change #1)
    merged = merged.sort_values(by="Sum of SurveyCount2 R2", ascending=False)
    
    metrics_with_subcols = [
        "Sum of SurveyCount",
        "Sum of SurveyCount2",
        "TCR%",
        "CSAT%",
        "Weightage (Sumproduct)",
    ]
    
    impact_metrics = [
        "Impact %",
        "Mix Shift Impact",
        "Score Impact",
    ]
    
    data_dict = {}
    
    # Dimension columns with empty second level
    for col in group_cols:
        data_dict[(col, "")] = merged[col]
    
    # Metrics with R1/R2/Diff subcolumns
    for m in metrics_with_subcols:
        data_dict[(m, "R1")] = merged[f"{m} R1"]
        data_dict[(m, "R2")] = merged[f"{m} R2"]
        data_dict[(m, "Diff")] = merged[f"{m} Diff"]
    
    # Impact metrics with single column (no subcolumn)
    for m in impact_metrics:
        data_dict[(m, "")] = merged[m]


    multi_df = pd.DataFrame(data_dict)
    multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns)
    
    # Create ONE styler object and chain ALL formatting + coloring
    def format_numeric(val):
        if pd.isna(val):
            return ""
        if isinstance(val, (int, float)):
            return "{:.2f}".format(val)
        return str(val)  # Strings/dims stay as-is
    
    # Robust detection of ALL "Diff" subcolumns + "Impact %" column in MultiIndex
    diff_cols_to_style = [col for col in multi_df.columns if col[1] == "Diff"]
    impact_cols_to_style = [col for col in multi_df.columns if col[0] == "Impact %"]
    all_cols_to_style = diff_cols_to_style + impact_cols_to_style
    
    def color_impact(val):
        if pd.isna(val):
            return ''
        elif val > 0:
            return 'background-color: #d4edda; color: #155724'
        elif val < 0:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    # Fixed: Target numeric columns only with IndexSlice
    numeric_metrics = ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]
    styled_multi_df = multi_df.style
    
    # Format numeric subcolumns
    for m in numeric_metrics:
        styled_multi_df = styled_multi_df.format(
            formatter=format_numeric,
            subset=i_slice[m, ["R1", "R2", "Diff"]]
        )
    # Format impact metrics
    styled_multi_df = styled_multi_df.format(
        formatter=format_numeric,
        subset=i_slice[["Impact %", "Mix Shift Impact", "Score Impact"], ""]
    )
    
    # Color impacts
    styled_multi_df = styled_multi_df.applymap(color_impact, subset=all_cols_to_style)

    
    st.subheader("Comparison Table 📚")
    st.dataframe(styled_multi_df)



    grand_total_1 = stats1.iloc[-1:]
    grand_total_2 = stats2.iloc[-1:]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='background-color:grey; padding:7px; font-weight:bold;'>"
            f"Grand Total - Range 1:<br>"
            f"SurveyCount: {grand_total_1['Sum of SurveyCount'].values[0]}<br>"
            f"TCR%: {grand_total_1['TCR%'].values[0]:.2%}<br>"
            f"CSAT%: {grand_total_1['CSAT%'].values[0]:.2%}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
            f"Grand Total - Range 2:<br>"
            f"SurveyCount: {grand_total_2['Sum of SurveyCount'].values[0]}<br>"
            f"TCR%: {grand_total_2['TCR%'].values[0]:.2%}<br>"
            f"CSAT%: {grand_total_2['CSAT%'].values[0]:.2%}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    total_impact = merged["Impact %"].sum()
    total_mix_shift = merged["Mix Shift Impact"].sum()
    total_score_impact = merged["Score Impact"].sum()

    col3, col4, col5 = st.columns(3)
    with col3:
        color = "#d4edda" if total_impact > 0 else "#f8d7da" if total_impact < 0 else "white"
        text_color = "#155724" if total_impact > 0 else "#721c24" if total_impact < 0 else "black"
        st.markdown(
            f"<div style='background-color:{color}; color:{text_color}; padding:7px; font-weight:bold; border:1px solid grey;'>"
            f"Total Impact %:<br>{total_impact:.4f}</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div style='background-color:grey; padding:7px; font-weight:bold; border:1px solid grey;'>"
            f"Total Mix Shift Impact:<br>{total_mix_shift:.4f}</div>",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"<div style='background-color:grey; padding:7px; font-weight:bold; border:1px solid grey;'>"
            f"Total Score Impact:<br>{total_score_impact:.4f}</div>",
            unsafe_allow_html=True,
        )


else:
    st.info("Upload an Excel file to get started.")
