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

            other_filters = {
                k: v for k, v in st.session_state.active_filters.items()
                if k != dim and v is not None
            }

            if other_filters:
                mask = pd.Series(True, index=df.index)
                for k, v in other_filters.items():
                    mask &= df[k].astype(str).str.strip().eq(str(v).strip())
            else:
                mask = pd.Series(True, index=df.index)

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

    # ---------------------- Fixed calc_group_stats ----------------------
    def calc_group_stats(dataframe, start_date, end_date, group_cols):
        mask = (dataframe["recordeddate"] >= pd.to_datetime(start_date)) & (dataframe["recordeddate"] <= pd.to_datetime(end_date))
        filtered = dataframe.loc[mask].copy()

        # Ensure numeric columns
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
            # Single-row total when no grouping
            grouped = pd.DataFrame([{
                "Sum of TCR_Yes": filtered["Sum of TCR_Yes"].sum(),
                "Sum of CSAT_Num": filtered["Sum of CSAT_Num"].sum(),
                "Sum of SurveyCount": filtered["Sum of SurveyCount"].sum(),
            }])

        # Compute additional metrics
        grouped["Sum of SurveyCount2"] = 0 if total_survey_count == 0 else grouped["Sum of SurveyCount"] / total_survey_count * 100
        grouped["TCR%"] = grouped["Sum of TCR_Yes"] * 100 / grouped["Sum of SurveyCount"].replace(0, 1)
        grouped["CSAT%"] = grouped["Sum of CSAT_Num"] * 100 / grouped["Sum of SurveyCount"].replace(0, 1)
        grouped["Weightage (Sumproduct)"] = (grouped["Sum of SurveyCount2"] / 100) * grouped["TCR%"]

        # Always return consistent column order
        cols = group_cols + ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]
        grouped = grouped[cols] if group_cols else grouped[["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]]

        return grouped
    # ---------------------------------------------------------------------

    active_keys = list(st.session_state.active_filters.keys())
    group_cols = active_keys if active_keys else []

    stats1 = calc_group_stats(filtered_df, date_range1[0], date_range1[1], group_cols)
    stats2 = calc_group_stats(filtered_df, date_range2[0], date_range2[1], group_cols)

    tab1, tab2 = st.tabs(["Comparison Table (Range1 vs Range2)", "Graph and Visualaisation"])

    with tab1:
        try:
            base = group_cols.copy()
            metrics = ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]

            filters_active = any(val is not None for val in st.session_state.active_filters.values())

            if filters_active:
                df_R1 = stats1[:-1].copy() if len(stats1) > 1 else stats1.copy()
                df_R2 = stats2[:-1].copy() if len(stats2) > 1 else stats2.copy()

                if base:
                    merged = df_R1.merge(df_R2, on=base, how="outer", suffixes=("_R1", "_R2"))
                else:
                    df_R1["dummy"] = 1
                    df_R2["dummy"] = 1
                    merged = df_R1.merge(df_R2, on="dummy", suffixes=("_R1", "_R2")).drop(columns=["dummy"])

                merged["Impact %"] = merged["Weightage (Sumproduct)_R2"] - merged["Weightage (Sumproduct)_R1"]

                w1 = df_R1.reset_index(drop=True)
                w2 = df_R2.reset_index(drop=True)
                for c in ["TCR%", "Sum of SurveyCount2", "Weightage (Sumproduct)"]:
                    if c in w1.columns: w1[c] = pd.to_numeric(w1[c], errors="coerce")
                    if c in w2.columns: w2[c] = pd.to_numeric(w2[c], errors="coerce")

                merged["Mix Shift Impact"] = ((w1["TCR%"] / 100) * w2["Sum of SurveyCount2"]).round(2)
                merged["Score Impact"] = ((w1["Sum of SurveyCount2"] / 100) * w2["TCR%"]).round(2)

                multi_cols = []
                for g in base: multi_cols.append((g, ""))
                for m in metrics:
                    multi_cols.append((m, "R1"))
                    multi_cols.append((m, "R2"))
                multi_cols += [("Impact %", ""), ("Mix Shift Impact", ""), ("Score Impact", "")]
                merged = merged.reindex(columns=[*base] +
                                                  [f"{m}_R1" for m in metrics] +
                                                  [f"{m}_R2" for m in metrics] +
                                                  ["Impact %", "Mix Shift Impact", "Score Impact"])
                merged.columns = pd.MultiIndex.from_tuples(multi_cols)

            else:
                # Default single-row total
                r1_tot = stats1.iloc[0]
                r2_tot = stats2.iloc[0]

                total_dict = {}
                if base:
                    total_dict.update({g: "Grand Total" for g in base})

                total_dict.update({
                    "Sum of SurveyCount_R1": r1_tot.get("Sum of SurveyCount", 0),
                    "Sum of SurveyCount_R2": r2_tot.get("Sum of SurveyCount", 0),
                    "Sum of SurveyCount2_R1": r1_tot.get("Sum of SurveyCount2", 0),
                    "Sum of SurveyCount2_R2": r2_tot.get("Sum of SurveyCount2", 0),
                    "TCR%_R1": r1_tot.get("TCR%", 0),
                    "TCR%_R2": r2_tot.get("TCR%", 0),
                    "CSAT%_R1": r1_tot.get("CSAT%", 0),
                    "CSAT%_R2": r2_tot.get("CSAT%", 0),
                    "Weightage (Sumproduct)_R1": r1_tot.get("Weightage (Sumproduct)", 0),
                    "Weightage (Sumproduct)_R2": r2_tot.get("Weightage (Sumproduct)", 0),
                    "Impact %": r2_tot.get("Weightage (Sumproduct)", 0) - r1_tot.get("Weightage (Sumproduct)", 0),
                    "Mix Shift Impact": ((r1_tot.get("TCR%", 0)/100) * r2_tot.get("Sum of SurveyCount2", 0)).round(2),
                    "Score Impact": ((r1_tot.get("Sum of SurveyCount2", 0)/100) * r2_tot.get("TCR%", 0)).round(2)
                })

                total_row = pd.DataFrame([total_dict])

                multi_cols = []
                for g in base: multi_cols.append((g, ""))
                for m in metrics:
                    multi_cols.append((m, "R1"))
                    multi_cols.append((m, "R2"))
                multi_cols += [("Impact %", ""), ("Mix Shift Impact", ""), ("Score Impact", "")]
                total_row = total_row.reindex(columns=[*base] +
                                                       [f"{m}_R1" for m in metrics] +
                                                       [f"{m}_R2" for m in metrics] +
                                                       ["Impact %", "Mix Shift Impact", "Score Impact"])
                total_row.columns = pd.MultiIndex.from_tuples(multi_cols)
                merged = total_row

            st.write("### Comparison (R1 vs R2 with Impact & Mix/Score)")
            st.dataframe(merged, use_container_width=True)

            # Grand Total Boxes
            colR1, colR2 = st.columns(2)
            with colR1:
                st.markdown("### **Grand Total — R1**")
                st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                            f"<b>Sum of SurveyCount:</b> {int(r1_tot.get('Sum of SurveyCount', 0))}<br>"
                            f"<b>TCR %:</b> {r1_tot.get('TCR%', 0):.2f}%<br>"
                            f"<b>CSAT %:</b> {r1_tot.get('CSAT%', 0):.2f}%<br></div>",
                            unsafe_allow_html=True)
            with colR2:
                st.markdown("### **Grand Total — R2**")
                st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                            f"<b>Sum of SurveyCount:</b> {int(r2_tot.get('Sum of SurveyCount', 0))}<br>"
                            f"<b>TCR %:</b> {r2_tot.get('TCR%', 0):.2f}%<br>"
                            f"<b>CSAT %:</b> {r2_tot.get('CSAT%', 0):.2f}%<br></div>",
                            unsafe_allow_html=True)

            # Total Impact Boxes
            tot_impact = merged["Impact %"].sum()
            tot_mix_shift = merged["Mix Shift Impact"].sum()
            tot_score_impact = merged["Score Impact"].sum()
            col3, col4, col5 = st.columns(3)
            with col3:
                st.markdown("### **Total Impact %**")
                st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                            f"Total Impact % : {tot_impact:.2f}<br></div>",
                            unsafe_allow_html=True)
            with col4:
                st.markdown("### **Total Mix Shift Impact**")
                st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                            f"Total : {tot_mix_shift:.2f}<br></div>",
                            unsafe_allow_html=True)
            with col5:
                st.markdown("### **Total Score Impact**")
                st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                            f"Total : {tot_score_impact:.2f}<br></div>",
                            unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Can't render comparison table: {e}")

else:
    st.info("Upload an Excel file to get started.")
