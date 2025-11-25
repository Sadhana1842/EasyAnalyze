import streamlit as st
import pandas as pd
from collections import OrderedDict

st.set_page_config(page_title="HP EasyAnalyze", layout="wide", page_icon="🧊")
st.title("HP EasyAnalyze")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    if not pd.api.types.is_datetime64_any_dtype(df["recordeddate"]):
        df["recordeddate"] = pd.to_datetime(df["recordeddate"], errors="coerce")

    min_date = df["recordeddate"].min()
    max_date = df["recordeddate"].max()

    st.sidebar.write("Please select date ranges")
    date_range1 = st.sidebar.date_input("Date Range 1", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    date_range2 = st.sidebar.date_input("Date Range 2", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    restricted_cols = [
        "recordeddate", "Sum of SurveyCount", "Sum of SurveyCount2",
        "Sum of TCR_Yes", "Sum of TCR_No", "Sum of CSAT_Num"
    ]
    available_dims = [col for col in df.columns if col not in restricted_cols]
    st.markdown("### Available Dimensions")

    if "active_filters" not in st.session_state:
        st.session_state.active_filters = OrderedDict()

    top_cols = st.columns(6)

    for i, col in enumerate(available_dims):
        if top_cols[i % 6].button(col, key=f"addbtn_{col}"):
            if col not in st.session_state.active_filters:
                st.session_state.active_filters[col] = None
                st.rerun()

    df = df.fillna("")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    if st.session_state.active_filters:
        st.sidebar.markdown("### Active Filters")

        for dim in list(st.session_state.active_filters.keys()):
            mask = pd.Series(True, index=df.index)
            for other_dim, other_val in st.session_state.active_filters.items():
                if other_val is not None and other_dim != dim:
                    mask &= df[other_dim].astype(str).str.strip().eq(str(other_val).strip())

            temp_df = df.loc[mask]
            possible_vals_raw = temp_df[dim].dropna().unique().tolist()
            possible_vals = sorted([str(x) for x in possible_vals_raw], key=lambda x: x.lower())

            current_val = None if st.session_state.active_filters[dim] is None else str(st.session_state.active_filters[dim])
            selected_val = st.sidebar.selectbox(
                f"{dim}",
                ["--Select--"] + possible_vals,
                index=(possible_vals.index(current_val) + 1) if current_val in possible_vals else 0,
                key=f"sb_{dim}"
            )

            st.session_state.active_filters[dim] = None if selected_val == "--Select--" else selected_val

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
            filtered_df = filtered_df[filtered_df[dim].astype(str).str.strip().eq(str(val).strip())]

    def calc_group_stats(dataframe, start_date, end_date, group_cols):
        mask = (dataframe["recordeddate"] >= pd.to_datetime(start_date)) & (dataframe["recordeddate"] <= pd.to_datetime(end_date))
        filtered = dataframe.loc[mask].copy()

        for col in ["Sum of SurveyCount", "Sum of TCR_Yes", "Sum of CSAT_Num"]:
            if col in filtered.columns:
                filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(0)

        total_survey_count = filtered["Sum of SurveyCount"].sum()

        if group_cols:
            grouped = (
                filtered.groupby(group_cols)
                .agg({
                    "Sum of TCR_Yes": "sum",
                    "Sum of CSAT_Num": "sum",
                    "Sum of SurveyCount": "sum",
                })
                .reset_index()
            )
        else:
            grouped = pd.DataFrame([{
                "Sum of TCR_Yes": filtered["Sum of TCR_Yes"].sum(),
                "Sum of CSAT_Num": filtered["Sum of CSAT_Num"].sum(),
                "Sum of SurveyCount": filtered["Sum of SurveyCount"].sum(),
            }])

        if total_survey_count == 0:
            grouped["Sum of SurveyCount2"] = 0
        else:
            grouped["Sum of SurveyCount2"] = grouped["Sum of SurveyCount"] / total_survey_count * 100

        grouped["TCR%"] = grouped["Sum of TCR_Yes"] * 100 / grouped["Sum of SurveyCount"].replace(0, 1)
        grouped["CSAT%"] = grouped["Sum of CSAT_Num"] * 100 / grouped["Sum of SurveyCount"].replace(0, 1)
        grouped["Weightage (Sumproduct)"] = ((grouped["Sum of SurveyCount2"] / 100) * grouped["TCR%"])

        if group_cols:
            total_row = pd.Series(
                {group_cols[0]: "Grand Total",
                 "Sum of SurveyCount": grouped["Sum of SurveyCount"].sum(),
                 "Sum of SurveyCount2": grouped["Sum of SurveyCount2"].sum(),
                 "TCR%": grouped["Sum of TCR_Yes"].sum() * 100 / grouped["Sum of SurveyCount"].sum(),
                 "CSAT%": grouped["Sum of CSAT_Num"].sum() * 100 / grouped["Sum of SurveyCount"].sum(),
                 "Weightage (Sumproduct)": None}
            )
        else:
            total_row = None

        cols = group_cols + ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]
        grouped = grouped[cols]

        if total_row is not None:
            grouped = pd.concat([grouped, total_row.to_frame().T], ignore_index=True)

        return grouped

    active_keys = list(st.session_state.active_filters.keys())
    group_cols = active_keys if active_keys else []

    stats1 = calc_group_stats(filtered_df, date_range1[0], date_range1[1], group_cols)
    stats2 = calc_group_stats(filtered_df, date_range2[0], date_range2[1], group_cols)

    tab1, tab2, tab3 = st.tabs(["Comparison Table", "Over all Impact Analysis", "Score and Mix Shift Impact Analysis"])

    # =====================================================================
    #                     TAB-1 — FIXED (YOUR REQUIRED FORMAT)
    # =====================================================================
    with tab1:
        try:
            base = group_cols.copy()

            metrics = ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]

            df_R1 = stats1.copy()
            df_R2 = stats2.copy()

            if base:
                merged = df_R1.merge(df_R2, on=base, how="outer", suffixes=("_R1", "_R2"))
            else:
                df_R1["dummy"] = 1
                df_R2["dummy"] = 1
                merged = df_R1.merge(df_R2, on="dummy", suffixes=("_R1", "_R2")).drop(columns=["dummy"])

            # Build MultiIndex
            new_cols = []

            for col in merged.columns:
                if col in base:
                    new_cols.append((col, ""))  # Main filter column has no R1/R2
                else:
                    orig = col.replace("_R1", "").replace("_R2", "")
                    period = "R1" if col.endswith("_R1") else "R2"
                    new_cols.append((orig, period))

            merged.columns = pd.MultiIndex.from_tuples(new_cols)

            st.write("### Comparison (R1 vs R2)")
            st.dataframe(merged, use_container_width=True)

        except Exception as e:
            st.error(f"Can't render comparison table: {e}")

    # =====================================================================
    # TABS 2 & 3 UNCHANGED — YOUR EXISTING LOGIC
    # =====================================================================

    with tab2:
        st.subheader("Impact %")
        w1 = stats1.iloc[:-1].reset_index(drop=True)
        w2 = stats2.iloc[:-1].reset_index(drop=True)

        merged = w1.merge(w2[group_cols + ["Weightage (Sumproduct)"]], on=group_cols, suffixes=("_1", "_2"))
        merged["Impact %"] = merged["Weightage (Sumproduct)_2"] - merged["Weightage (Sumproduct)_1"]
        st.dataframe(merged[group_cols + ["Impact %"]].reset_index(drop=True))
        total_weight = merged["Impact %"].sum()
        st.markdown(
            f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
            f"Grand Total:<br>Total Impact % : {total_weight:.2f} %<br></div>",
            unsafe_allow_html=True,
        )

    with tab3:
        st.header("Mix-shift and Score Impact Analysis")
        w1 = stats1.iloc[:-1].reset_index(drop=True)
        w2 = stats2.iloc[:-1].reset_index(drop=True)

        cols_to_num = ["TCR%", "CSAT%", "Sum of SurveyCount2", "Weightage (Sumproduct)"]
        for df_ in [w1, w2]:
            for c in cols_to_num:
                if c in df_.columns:
                    df_[c] = pd.to_numeric(df_[c], errors="coerce")

        w1["Mix Shift Impact"] = ((w1["TCR%"] / 100) * w2["Sum of SurveyCount2"]).round(2)
        w1["Score Impact"] = ((w1["Sum of SurveyCount2"] / 100) * w2["TCR%"]).round(2)

        tot_ms_impact = w1["Mix Shift Impact"].sum()
        tot_sc_impact = w1["Score Impact"].sum()
        tcr_total1 = stats1.iloc[-1]["TCR%"]

        tot_impact_ms = tot_ms_impact - tcr_total1
        tot_impact_sc = tot_sc_impact - tcr_total1
        denom = (tot_impact_ms + tot_impact_sc) if (tot_impact_ms + tot_impact_sc) != 0 else 1
        ms_impact_ratio = tot_impact_ms / denom
        sc_impact_ratio = tot_impact_sc / denom

        col1, col2 = st.columns(2, border=True)
        with col1:
            st.subheader("Mix Shift Impact")
            st.dataframe(w1[group_cols + ["Mix Shift Impact"]])
            st.markdown(
                f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                f"Total : {tot_ms_impact:.4f}<br>"
                f"Total Impact : {tot_impact_ms:.4f}<br>"
                f"Impact Ratio : {ms_impact_ratio:.4%}<br></div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.subheader("Score Impact")
            st.dataframe(w1[group_cols + ["Score Impact"]])
            st.markdown(
                f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                f"Total : {tot_sc_impact:.4f}<br>"
                f"Total Impact : {tot_impact_sc:.4f}<br>"
                f"Impact Ratio : {sc_impact_ratio:.4%}<br></div>",
                unsafe_allow_html=True,
            )

else:
    st.info("Upload an Excel file to get started.")
