def calc_group_stats(dataframe, start_date, end_date, group_cols):
    mask = (dataframe["recordeddate"] >= pd.to_datetime(start_date)) & (dataframe["recordeddate"] <= pd.to_datetime(end_date))
    filtered = dataframe.loc[mask].copy()

    for col in ["Sum of SurveyCount", "Sum of TCR_Yes", "Sum of CSAT_Num"]:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(0)

    total_survey_count = filtered["Sum of SurveyCount"].sum()

    # Grouped data
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
        # No group columns → create single row with placeholder for group_cols
        grouped = pd.DataFrame([{
            col: "Grand Total" for col in group_cols
        } if group_cols else {}])
        # Add metrics
        grouped["Sum of SurveyCount"] = total_survey_count
        grouped["Sum of TCR_Yes"] = filtered["Sum of TCR_Yes"].sum()
        grouped["Sum of CSAT_Num"] = filtered["Sum of CSAT_Num"].sum()

    # Calculate derived metrics
    grouped["Sum of SurveyCount2"] = 0 if total_survey_count == 0 else grouped["Sum of SurveyCount"] / total_survey_count * 100
    grouped["TCR%"] = grouped["Sum of TCR_Yes"] * 100 / grouped["Sum of SurveyCount"].replace(0, 1)
    grouped["CSAT%"] = grouped["Sum of CSAT_Num"] * 100 / grouped["Sum of SurveyCount"].replace(0, 1)
    grouped["Weightage (Sumproduct)"] = (grouped["Sum of SurveyCount2"] / 100) * grouped["TCR%"]

    # Ensure consistent column order
    final_cols = group_cols + ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"] if group_cols else ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]
    grouped = grouped[final_cols]

    return grouped

# -------------------- TAB1 --------------------
tab1, tab2 = st.tabs(["Comparison Table (Range1 vs Range2)", "Graph and Visualaisation"])

with tab1:
    try:
        metrics = ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]

        # Prepare merged table if there are active group_cols
        if group_cols:
            df_R1 = stats1.copy()
            df_R2 = stats2.copy()
            merged = df_R1.merge(df_R2, on=group_cols, how="outer", suffixes=("_R1", "_R2"))

            # Calculations
            merged["Impact %"] = merged["Weightage (Sumproduct)_R2"] - merged["Weightage (Sumproduct)_R1"]

            for c in ["TCR%", "Sum of SurveyCount2", "Weightage (Sumproduct)"]:
                merged[f"{c}_R1"] = pd.to_numeric(merged[f"{c}_R1"], errors="coerce")
                merged[f"{c}_R2"] = pd.to_numeric(merged[f"{c}_R2"], errors="coerce")

            merged["Mix Shift Impact"] = ((merged["TCR%_R1"] / 100) * merged["Sum of SurveyCount2_R2"]).round(2)
            merged["Score Impact"] = ((merged["Sum of SurveyCount2_R1"] / 100) * merged["TCR%_R2"]).round(2)

            # MultiIndex columns
            multi_cols = []
            for g in group_cols: multi_cols.append((g, ""))
            for m in metrics:
                multi_cols.append((m, "R1"))
                multi_cols.append((m, "R2"))
            multi_cols += [("Impact %", ""), ("Mix Shift Impact", ""), ("Score Impact", "")]
            merged = merged.reindex(columns=group_cols + [f"{m}_R1" for m in metrics] + [f"{m}_R2" for m in metrics] + ["Impact %", "Mix Shift Impact", "Score Impact"])
            merged.columns = pd.MultiIndex.from_tuples(multi_cols)

            st.write("### Comparison (R1 vs R2 with Impact & Mix/Score)")
            st.dataframe(merged, use_container_width=True)

        # Always show Markdown totals
        r1_tot = stats1.iloc[0]
        r2_tot = stats2.iloc[0]
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

        tot_impact = r2_tot["Weightage (Sumproduct)"] - r1_tot["Weightage (Sumproduct)"]
        tot_mix_shift = ((r1_tot["TCR%"] / 100) * r2_tot["Sum of SurveyCount2"]).round(2)
        tot_score_impact = ((r1_tot["Sum of SurveyCount2"] / 100) * r2_tot["TCR%"]).round(2)
        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown("### **Total Impact %**")
            st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                        f"Total Impact % : {tot_impact:.2f}<br></div>", unsafe_allow_html=True)
        with col4:
            st.markdown("### **Total Mix Shift Impact**")
            st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                        f"Total : {tot_mix_shift:.2f}<br></div>", unsafe_allow_html=True)
        with col5:
            st.markdown("### **Total Score Impact**")
            st.markdown(f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                        f"Total : {tot_score_impact:.2f}<br></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Can't render comparison table: {e}")
