import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import plotly.express as px
import tempfile

def weighted_mean(x, w):
    return round(np.average(x, weights=w), 5)

def weighted_median(x, w):
    sorted_idx = np.argsort(x)
    x_sorted = np.array(x)[sorted_idx]
    w_sorted = np.array(w)[sorted_idx]
    cum_w = np.cumsum(w_sorted)
    cutoff = sum(w_sorted) / 2.0
    return round(x_sorted[np.where(cum_w >= cutoff)[0][0]], 5)

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".sav"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        df, _ = pyreadstat.read_sav(tmp_path)
        return df
    else:
        st.error("Geçersiz dosya türü.")
        return None

def weighted_group_analysis(df, group, weight, na_th):
    df = df.loc[:, df.isna().mean() < na_th].copy()
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    text_vars = df.select_dtypes(include=['object', 'string']).columns.tolist()

    if not weight or weight not in df.columns:
        df['Weight_Fallback'] = 1
        weight = 'Weight_Fallback'

    results = {}
    for var in numeric_vars:
        if var != weight:
            temp = df[group + [var, weight]].dropna()
            grouped = temp.groupby(group).apply(lambda g: pd.Series({
                'weighted_mean': weighted_mean(g[var], g[weight]),
                'weighted_median': weighted_median(g[var], g[weight])
            })).reset_index()
            results[var] = grouped

    for var in text_vars:
        if var not in group:
            temp = df[group + [var, weight]].dropna()
            temp["words"] = temp[var].astype(str).str.split()
            temp = temp.explode("words")
            grouped = temp.groupby(group + ["words"])[weight].sum().reset_index(name="weighted_count")
            grouped['total'] = grouped.groupby(group)["weighted_count"].transform('sum')
            grouped['frequency'] = grouped['weighted_count'] / grouped['total']
            results[var] = grouped.sort_values("weighted_count", ascending=False)

    return results


st.set_page_config(layout="wide")
st.title("📊 Weighted Group Analysis")

uploaded_file = st.file_uploader("Veri dosyası yükle (.csv, .xlsx, .sav)", type=["csv", "xlsx", "xls", "sav"])

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        with st.form("analysis_form"):
            group_col = st.multiselect("Gruplama Değişken(ler)i", df.select_dtypes(include=['object', 'category']).columns)
            weight_col = st.selectbox("Ağırlık Değişkeni", [''] + df.select_dtypes(include=[np.number]).columns.tolist())
            na_th = st.slider("NA Eşiği", 0.0, 1.0, 0.5, step=0.05)

            selected_vars = []
            plot_type = None

            if group_col:
                dummy_result = weighted_group_analysis(df, group_col, weight_col, na_th)
                selected_vars = st.multiselect("Analiz Değişkenleri", list(dummy_result.keys()))
                plot_type = st.radio("Grafik Tipi", ["bar", "line", "box", "point", "pie", "density"])

            submit = st.form_submit_button("Analizi Başlat")

        if submit and group_col and selected_vars:
            result = weighted_group_analysis(df, group_col, weight_col, na_th)
            for var in selected_vars:
                st.subheader(f"📈 {var}")
                data = result[var]

                if 'frequency' in data.columns:
                    top_words = data.groupby("words")["weighted_count"].sum().nlargest(10).index
                    filtered = data[data["words"].isin(top_words)]

                    if plot_type == "bar":
                        fig = px.bar(filtered, x="words", y="frequency", color=group_col[0],
                                     facet_col=group_col[1] if len(group_col) > 1 else None)
                    elif plot_type == "pie":
                        if len(group_col) > 1:
                            for val in filtered[group_col[1]].unique():
                                subset = filtered[filtered[group_col[1]] == val]
                                fig = px.pie(subset, values="frequency", names="words",
                                             title=f"{group_col[1]}: {val}")
                                st.plotly_chart(fig, use_container_width=True)
                            continue
                        else:
                            fig = px.pie(filtered, values="frequency", names="words", title=f"{var} - Pie")
                    elif plot_type == "density":
                        fig = px.violin(filtered, y="frequency", color=group_col[0], box=True, points="all")

                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(filtered)

                else:
                    long_df = pd.melt(data, id_vars=group_col, value_vars=['weighted_mean', 'weighted_median'])

                    if plot_type == "bar":
                        fig = px.bar(long_df, x=group_col[0], y="value", color="variable", barmode="group",
                                     facet_col=group_col[1] if len(group_col) > 1 else None)
                    elif plot_type == "line":
                        fig = px.line(data, x=group_col[0], y="weighted_mean",
                                      color=group_col[1] if len(group_col) > 1 else None, markers=True)
                    elif plot_type == "box":
                        fig = px.box(data, x=group_col[0], y="weighted_mean",
                                     color=group_col[1] if len(group_col) > 1 else None)
                    elif plot_type == "point":
                        fig = px.scatter(data, x=group_col[0], y="weighted_mean",
                                         color=group_col[1] if len(group_col) > 1 else None)

                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(data)
