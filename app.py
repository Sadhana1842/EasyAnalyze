import streamlit as st
import pandas as pd
from collections import OrderedDict

st.set_page_config(page_title="HP EasyAnalyze", layout="wide", page_icon="🧊")
st.title("HP EasyAnalyze")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # Ensure recordeddate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["recordeddate"]):
        df["recordeddate"] = pd.to_datetime(df["recordeddate"], errors="coerce")

    min_date = df["recordeddate"].min()
    max_date = df["recordeddate"].max()

    st.sidebar.write("Please select date ranges")
    date_range1 = st.sidebar.date_input("Date Range 1", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    date_range2 = st.sidebar.date_input("Date Range 2", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # ---------------- FILTERING LOGIC START ----------------
    st.sidebar.markdown("### 🔍 Add Filters")

    restricted_cols = [
        "recordeddate", "Sum of SurveyCount", "Sum of SurveyCount2",
        "Sum of TCR_Yes", "Sum of TCR_No", "Sum of CSAT_Num"
    ]
    available_dims = [col for col in df.columns if col not in restricted_cols]
    st.markdown("### Available Dimensions")

    # ---------------------------
    # INITIALIZE session_state BEFORE using it
    # ---------------------------
    if "active_filters" not in st.session_state:
        st.session_state.active_filters = OrderedDict()

    # 6 columns across the top (auto-wrap to next row)
    top_cols = st.columns(6)

    for i, col in enumerate(available_dims):
        if top_cols[i % 6].button(col, key=f"addbtn_{col}"):
            if col not in st.session_state.active_filters:
                st.session_state.active_filters[col] = None
                st.rerun()

    # Clean string columns globally for robust comparison
    df = df.fillna("")  # replace NaN with empty string
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    #df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)  # optional for case-insensitive filtering

    # Build dependent selectboxes (value lists come from the df filtered by *other* active filters)
    if st.session_state.active_filters:
        st.sidebar.markdown("### Active Filters")

        for dim in list(st.session_state.active_filters.keys()):
            # Build mask by applying all other active filters
            mask = pd.Series(True, index=df.index)
            for other_dim, other_val in st.session_state.active_filters.items():
                if other_val is not None and other_dim != dim:
                    if isinstance(other_val, list):
                        mask &= df[other_dim].astype(str).str.strip().isin([str(v).strip() for v in other_val])
                    else:
                        mask &= df[other_dim].astype(str).str.strip().eq(str(other_val).strip())

            temp_df = df.loc[mask]

            # Get unique values for THIS column only from the already-filtered subset
            possible_vals_raw = temp_df[dim].dropna().unique().tolist()
            possible_vals = sorted([str(x) for x in possible_vals_raw], key=lambda x: x.lower())

            current_val = None if st.session_state.active_filters[dim] is None else str(st.session_state.active_filters[dim])

            # single-select
            selected_val = st.sidebar.selectbox(
                f"{dim}",
                ["--Select--"] + possible_vals,
                index=(possible_vals.index(current_val) + 1) if current_val in possible_vals else 0,
                key=f"sb_{dim}"
            )

            # Store None if placeholder chosen; otherwise store the real value
            st.session_state.active_filters[dim] = None if (selected_val == "--Select--") else selected_val

            # Reset / Remove buttons
            reset_col, remove_col = st.sidebar.columns(2)
            if reset_col.button(f"🔁 Reset {dim}", key=f"reset_{dim}"):
                st.session_state.active_filters[dim] = None
                st.rerun()
            if remove_col.button(f"❌ Remove {dim}", key=f"remove_{dim}"):
                del st.session_state.active_filters[dim]
                st.rerun()
    else:
        st.sidebar.info("No dimensions selected. Click a dimension above to add it as a filter.")

    # Apply all active filters to make the final filtered_df
    filtered_df = df.copy()
    for dim, val in st.session_state.active_filters.items():
        if val is not None:
            filtered_df = filtered_df[filtered_df[dim].astype(str).str.strip().eq(str(val).strip())]
    # ---------------- FILTERING LOGIC END ----------------


    # --- calculations (kept intact but with safe numeric coercion to prevent NoneType math errors) ---
    def calc_group_stats(dataframe, start_date, end_date, group_cols):
        mask = (dataframe["recordeddate"] >= pd.to_datetime(start_date)) & (dataframe["recordeddate"] <= pd.to_datetime(end_date))
        filtered = dataframe.loc[mask].copy()

        # SAFE COERCION: ensure numeric columns are numeric and NaNs -> 0 (does NOT change formulas, only avoids type errors)
        for col in ["Sum of SurveyCount", "Sum of TCR_Yes", "Sum of CSAT_Num"]:
            if col in filtered.columns:
                filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(0)

        total_survey_count = filtered["Sum of SurveyCount"].sum()

        grouped = (
            filtered.groupby(group_cols)
            .agg({
                "Sum of TCR_Yes": "sum",
                "Sum of CSAT_Num": "sum",
                "Sum of SurveyCount": "sum",
            })
            .reset_index()
        )

        # avoid division by zero
        if total_survey_count == 0:
            grouped["Sum of SurveyCount2"] = 0.0
        else:
            grouped["Sum of SurveyCount2"] = grouped["Sum of SurveyCount"] / total_survey_count * 100
        grouped["Sum of SurveyCount2"] = grouped["Sum of SurveyCount2"].round(2)

        grouped["TCR%"] = grouped.apply(lambda r: (r["Sum of TCR_Yes"] * 100.0 / r["Sum of SurveyCount"]) if r["Sum of SurveyCount"] else 0.0, axis=1)
        grouped["CSAT%"] = grouped.apply(lambda r: (r["Sum of CSAT_Num"] * 100.0 / r["Sum of SurveyCount"]) if r["Sum of SurveyCount"] else 0.0, axis=1)
        grouped["Weightage (Sumproduct)"] = ((grouped["Sum of SurveyCount2"] / 100) * grouped["TCR%"]).round(4)

        total_row = pd.Series(
            {
                group_cols[0]: "Grand Total",
                "Sum of SurveyCount": grouped["Sum of SurveyCount"].sum(),
                "Sum of SurveyCount2": grouped["Sum of SurveyCount2"].sum(),
                "TCR%": (grouped["Sum of TCR_Yes"].sum() / grouped["Sum of SurveyCount"].sum()) if grouped["Sum of SurveyCount"].sum() else 0.0,
                "CSAT%": (grouped["Sum of CSAT_Num"].sum() / grouped["Sum of SurveyCount"].sum()) if grouped["Sum of SurveyCount"].sum() else 0.0,
            }
        )

        grouped = grouped[group_cols + ["Sum of SurveyCount", "Sum of SurveyCount2", "TCR%", "CSAT%", "Weightage (Sumproduct)"]]
        grouped = pd.concat([grouped, total_row.to_frame().T], ignore_index=True)
        return grouped

    try:
        # If no filters are active, calculate overall stats (no grouping)
        active_keys = list(st.session_state.active_filters.keys())
        if active_keys:
            group_cols = active_keys
        else:
            group_cols = []  # No grouping → aggregate entire table

        stats1 = calc_group_stats(filtered_df, date_range1[0], date_range1[1], group_cols)
        stats2 = calc_group_stats(filtered_df, date_range2[0], date_range2[1], group_cols)
    except Exception as e:
        st.error(f"Can't calculate tables: {e}")

    tab1, tab2, tab3 = st.tabs(["Comparison Table", "Over all Impact Analysis", "Score and Mix Shift Impact Analysis"])
    with tab1:
        try:
            # --- BUILD MERGED MULTIINDEX DATAFRAME BASED ON SELECTED FILTER ---
            filter_col = selected_filter  # Example: "template", "TemplateCategory", etc.
            
            # R1 and R2 final grouped results (already calculated)
            df_R1 = result1.reset_index()
            df_R2 = result2.reset_index()
            
            # Rename columns to prepare for merge
            df_R1 = df_R1.add_suffix(" R1")
            df_R2 = df_R2.add_suffix(" R2")
            
            # Rename merge key back to normal for join
            df_R1 = df_R1.rename(columns={f"{filter_col} R1": filter_col})
            df_R2 = df_R2.rename(columns={f"{filter_col} R2": filter_col})
            
            # Merge based on actual filter value (IMPORTANT)
            merged = pd.merge(df_R1, df_R2, on=filter_col, how="outer")
            
            # Build MultiIndex columns → Main col + (R1/R2)
            new_cols = []
            for col in merged.columns:
                if col == filter_col:
                    new_cols.append((col, ""))   # filter column has no R1/R2
                elif col.endswith(" R1"):
                    new_cols.append((col.replace(" R1", ""), "R1"))
                elif col.endswith(" R2"):
                    new_cols.append((col.replace(" R2", ""), "R2"))
                else:
                    new_cols.append((col, ""))
            
            merged.columns = pd.MultiIndex.from_tuples(new_cols)
            
            # Show merged dataframe
            st.dataframe(merged)

            # ---- Grand Total Block (UNCHANGED) ----
            st.write("### Grand Total Summary")
            col1, col2 = st.columns(2, border=True)
            
            with col1:
                grand_total_1 = df1.iloc[-1:]
                st.markdown(
                    f"<div style='background-color:grey; padding:7px; font-weight:bold;'>"
                    f"Grand Total:<br>"
                    f"SurveyCount: {grand_total_1['Sum of SurveyCount'].values[0]}<br>"
                    f"TCR%: {grand_total_1['TCR%'].values[0]:.2%}<br>"
                    f"CSAT%: {grand_total_1['CSAT%'].values[0]:.2%}</div>",
                    unsafe_allow_html=True,
                )
            
            with col2:
                grand_total_2 = df2.iloc[-1:]
                st.markdown(
                    f"<div style='background-color:grey; padding:10px; font-weight:bold;'>"
                    f"Grand Total:<br>"
                    f"SurveyCount: {grand_total_2['Sum of SurveyCount'].values[0]}<br>"
                    f"TCR%: {grand_total_2['TCR%'].values[0]:.2%}<br>"
                    f"CSAT%: {grand_total_2['CSAT%'].values[0]:.2%}</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.error("Can't find dataset. Please upload file!")

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








