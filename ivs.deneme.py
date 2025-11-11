import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import plotly.express as px
import tempfile

# Yeni importlar
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
from sklearn.metrics import r2_score


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
        st.error("Invalid file type.")
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
st.title("📊 Inter-Sectoral Vulnerability Survey & Targeting")

uploaded_file = st.file_uploader("Upload data file (.csv, .xlsx, .sav)", type=["csv", "xlsx", "xls", "sav"])
st.markdown("## 🔔 Data Validation & File Processing Notice")
st.markdown("**Dear User,**")
st.markdown("""
We confirm that the display and labelling processes for the **rCSI, LCSI, and Severity Indicators** presented within our application have been completed in accordance with the highest standards of accuracy.
""")

st.markdown("### Important File Upload Information")
st.markdown("""
This rigorous process ensures the reliability and transparency of the data we provide to you.

***

Should you require further information regarding data interpretation, methodology, or file handling issues, 
please contact our support department.
""")

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        candidates = numeric_cols + cat_cols

        # Four tabs (Severity placeholder included)
        tab_basic, tab_advanced, tab_target, tab_severity = st.tabs(["1) Basic Analysis", "2) Advanced Analyses", "3) Targeting", "4) Severity"])

        # ------------------------------
        # 1) BASIC ANALYSIS
        # ------------------------------
        with tab_basic:
            st.header("Basic Analysis - Weighted Mean/Median/Frequency")
            analysis_vars = st.multiselect("Analysis variables", candidates)
            weight_col = st.selectbox("Weight (optional)", [''] + numeric_cols)
            group_choices = df.select_dtypes(include=['object', 'category']).columns.tolist()
            group_col = st.multiselect("Grouping (optional)", group_choices)

            if "run_basic" not in st.session_state:
                st.session_state["run_basic"] = False
            if st.button("Run analysis", key="basic_run"):
                st.session_state["run_basic"] = True

            if st.session_state["run_basic"]:
                if not analysis_vars:
                    st.warning("Select at least one variable.")
                else:
                    df_filtered = df.copy()

                    if not group_col:
                        tmp = "__ALL__"
                        df_filtered[tmp] = "All"
                        group_param = [tmp]
                        use_tmp = True
                    else:
                        group_param = group_col
                        use_tmp = False

                    results = weighted_group_analysis(df_filtered, group_param, weight_col, na_th=1.0)

                    tabs = st.tabs(analysis_vars)
                    for tab, var in zip(tabs, analysis_vars):
                        with tab:
                            data = results.get(var)
                            if data is None or data.empty:
                                st.info("No results")
                                continue

                            # Categorical -> weighted frequency
                            if var in cat_cols and 'words' in data.columns:
                                top_words = data.groupby("words")["weighted_count"].sum().nlargest(10).index
                                d = data[data["words"].isin(top_words)].copy()

                                color_arg = group_param[0] if len(group_param) >= 1 else None
                                facet_arg = group_param[1] if len(group_param) > 1 else None

                                fig = px.bar(
                                    d,
                                    x="words",
                                    y="frequency",
                                    color=color_arg,
                                    facet_col=facet_arg,
                                    barmode="group",
                                    title=f"{var} - weighted frequency"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.dataframe(d.sort_values(["weighted_count"], ascending=False), hide_index=True)

                            # Numeric -> weighted mean / median
                            elif var in numeric_cols and ('weighted_mean' in data.columns or 'weighted_median' in data.columns):
                                vals = [c for c in ['weighted_mean', 'weighted_median'] if c in data.columns]
                                long_df = pd.melt(data, id_vars=group_param, value_vars=vals)
                                facet_arg = group_param[1] if len(group_param) > 1 else None

                                fig = px.bar(
                                    long_df,
                                    x=group_param[0],
                                    y="value",
                                    color="variable",
                                    barmode="group",
                                    facet_col=facet_arg,
                                    title=f"{var} - weighted statistics"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.dataframe(data, hide_index=True)

                            else:
                                st.dataframe(data, hide_index=True)

                    if use_tmp:
                        df_filtered.drop(columns=[tmp], inplace=True)

        # ------------------------------
        # 2) ADVANCED ANALYSES
        # ------------------------------
        with tab_advanced:
            st.header("Advanced Analyses - Regression, Correlation, Chi-Square, Clustering")
            methods = st.multiselect("Select analyses to run", ["Regression", "Correlation", "Chi-Square", "Clustering"])

            if "Regression" in methods:
                st.subheader("Regression")
                reg_dep = st.selectbox("Dependent variable (numeric or binary categorical)", [''] + numeric_cols + cat_cols, key="reg_dep")
                reg_indep = st.multiselect("Independent variables (numeric)", numeric_cols, key="reg_indep")
                reg_model_type = st.selectbox("Model type", ["LinearRegression", "RandomForest", "RobustRegression", "LogisticRegression"], index=0, key="reg_model_type")
                if st.button("Run regression", key="run_reg"):
                    if not reg_dep:
                        st.error("Please select a dependent variable.")
                    elif not reg_indep:
                        st.error("Select at least one independent variable.")
                    else:
                        X = df[reg_indep].select_dtypes(include=[np.number]).dropna()
                        y_raw = df[reg_dep].loc[X.index]
                        if X.empty or y_raw.empty:
                            st.error("No data available for selected variables.")
                        else:
                            # Logistic requires binary target
                            if reg_model_type == "LogisticRegression":
                                y = y_raw.astype(str).dropna()
                                uniq = y.unique()
                                if len(uniq) > 2:
                                    st.error("Dependent variable must have two classes for logistic regression.")
                                    st.stop()
                                if len(uniq) == 1:
                                    st.error("Dependent variable has only one class; logistic regression cannot be applied.")
                                    st.stop()
                                y = (y == uniq[1]).astype(int)
                            else:
                                y = pd.to_numeric(y_raw, errors='coerce')
                                if y.isna().all():
                                    st.error("Dependent variable is not numeric or could not be converted.")
                                    st.stop()

                            # Split
                            X_train, X_test, y_train, y_test = train_test_split(X.loc[y.index], y, test_size=0.2, random_state=42)
                            # Scale for linear/robust/logistic
                            if reg_model_type in ("LinearRegression", "RobustRegression", "LogisticRegression"):
                                scaler = StandardScaler()
                                X_train_s = scaler.fit_transform(X_train)
                                X_test_s = scaler.transform(X_test)

                            if reg_model_type == "LinearRegression":
                                model = LinearRegression().fit(X_train_s, y_train)
                                preds = model.predict(X_test_s)
                                coefs = pd.Series(model.coef_, index=X.columns)
                                st.write("R2:", round(r2_score(y_test, preds), 4))
                                st.dataframe(coefs.rename("coef"))
                            elif reg_model_type == "RobustRegression":
                                huber = HuberRegressor().fit(X_train_s, y_train)
                                preds = huber.predict(X_test_s)
                                coefs = pd.Series(huber.coef_, index=X.columns)
                                st.write("R2 (approx):", round(r2_score(y_test, preds), 4))
                                st.dataframe(coefs.rename("coef"))
                            elif reg_model_type == "RandomForest":
                                rf = RandomForestRegressor(random_state=42, n_estimators=100).fit(X_train, y_train)
                                preds = rf.predict(X_test)
                                importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                                st.write("R2:", round(r2_score(y_test, preds), 4))
                                st.dataframe(importances.rename("importance"))
                            else:  # LogisticRegression
                                log = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
                                probs = log.predict_proba(X_test_s)[:, 1]
                                preds_class = log.predict(X_test_s)
                                acc = (preds_class == y_test).mean()
                                st.write("Accuracy (test):", round(acc, 4))
                                coefs = pd.Series(log.coef_.flatten(), index=X.columns)
                                st.dataframe(coefs.rename("coef"))

            if "Correlation" in methods:
                st.subheader("Correlation")
                corr_vars = st.multiselect("Numeric variables for correlation", numeric_cols, default=numeric_cols[:5], key="corr_vars")
                if st.button("Calculate correlation", key="run_corr"):
                    if len(corr_vars) < 2:
                        st.error("Select at least 2 numeric variables.")
                    else:
                        corr_df = df[corr_vars].corr().round(3)
                        fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Matrix")
                        st.plotly_chart(fig)
                        st.dataframe(corr_df)

            if "Chi-Square" in methods:
                st.subheader("Chi-Square Test")
                chi_x = st.selectbox("Category X", [''] + cat_cols, key="chi_x")
                chi_y = st.selectbox("Category Y", [''] + cat_cols, key="chi_y")
                if st.button("Run Chi-Square", key="run_chi"):
                    if not chi_x or not chi_y:
                        st.error("Select two categorical variables.")
                    else:
                        ct = pd.crosstab(df[chi_x].astype(str), df[chi_y].astype(str))
                        if (ct.values < 5).sum() > 0:
                            st.warning("Expected frequencies in contingency table may be small; interpret results with caution.")
                        chi2, p, dof, ex = chi2_contingency(ct)
                        st.write(f"chi2 = {chi2:.3f}, p-value = {p:.5f}, dof = {dof}")
                        st.dataframe(ct)

            if "Clustering" in methods:
                st.subheader("Clustering (KMeans)")
                cluster_vars = st.multiselect("Numeric variables for clustering", numeric_cols, default=numeric_cols[:3], key="cluster_vars")
                n_clusters = st.number_input("Number of clusters", min_value=2, max_value=20, value=3, key="n_clusters")
                if st.button("Create clusters", key="run_cluster"):
                    if len(cluster_vars) < 1:
                        st.error("Select at least 1 numeric variable.")
                    else:
                        Xc = df[cluster_vars].dropna()
                        if Xc.empty:
                            st.error("Not enough data for selected variables.")
                        else:
                            scaler = StandardScaler()
                            Xs = scaler.fit_transform(Xc)
                            kmeans = KMeans(n_clusters=int(n_clusters), random_state=42).fit(Xs)
                            Xc['cluster'] = kmeans.labels_
                            st.write("Observations per cluster:")
                            st.dataframe(Xc['cluster'].value_counts().rename_axis('cluster').reset_index(name='count'))
                            centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_vars)
                            st.write("Centroids (values):")
                            st.dataframe(centroids)

                            if Xs.shape[1] >= 2:
                                pca = PCA(n_components=2)
                                coords = pca.fit_transform(Xs)
                                scatter = pd.DataFrame(coords, columns=['PC1','PC2'])
                                scatter['cluster'] = kmeans.labels_
                                fig = px.scatter(scatter, x='PC1', y='PC2', color='cluster', title='Clustering PCA Visualization')
                                st.plotly_chart(fig)

        # ------------------------------
        # 3) TARGETING
        # ------------------------------
        with tab_target:
            st.header("Targeting - Identify poorest households via multi-dimensional poverty")
            st.markdown("Select variables to build the poverty index (they will be normalized and weighted).")
            poverty_vars = st.multiselect("Poverty indicator variables (numeric)", numeric_cols, key="poverty_vars")
            weights_input = st.text_input("Weights (comma-separated; should sum to 1). Leave empty for equal weights.", key="poverty_weights")
            dep_var = st.selectbox("Dependent variable for regression (optional; usually poverty index)", [''] + numeric_cols + cat_cols, key="target_dep")
            indep_vars = st.multiselect("Independent variables for regression (numeric or categorical)", candidates, key="target_indep")
            model_choice = st.selectbox("Model choice", ["LinearRegression", "RandomForest", "RobustRegression", "LogisticRegression"], key="target_model")
            top_n = st.number_input("How many poorest households to show", min_value=1, max_value=500, value=20, key="top_n")

            if st.button("Run targeting", key="run_target"):
                if not poverty_vars and not dep_var:
                    st.error("Please select at least one poverty indicator or a dependent variable.")
                else:
                    used_df = df.copy()
                    if poverty_vars:
                        if weights_input.strip():
                            try:
                                w = [float(x.strip()) for x in weights_input.split(",")]
                                if len(w) != len(poverty_vars):
                                    st.error("Number of weights does not match number of selected variables.")
                                    st.stop()
                                if abs(sum(w) - 1.0) > 1e-6:
                                    st.error("Weights must sum to 1.")
                                    st.stop()
                                weights = np.array(w)
                            except Exception:
                                st.error("Weights format is invalid.")
                                st.stop()
                        else:
                            weights = np.ones(len(poverty_vars)) / len(poverty_vars)

                        sub = used_df[poverty_vars].apply(pd.to_numeric, errors='coerce').copy()
                        sub = sub.dropna()
                        if sub.empty:
                            st.error("Poverty indicators do not contain sufficient numeric data.")
                            st.stop()

                        sub_n = (sub - sub.min()) / (sub.max() - sub.min()).replace(0, 1)
                        pi = sub_n.dot(weights)
                        used_df.loc[sub_n.index, 'poverty_index'] = pi
                        target_column = 'poverty_index'
                        st.write("Poverty index created. Example:")
                        st.dataframe(used_df[['poverty_index']].dropna().head())
                    else:
                        target_column = dep_var

                    if not indep_vars:
                        st.warning("No independent variables selected for regression. Only the poverty list will be shown.")
                        poorest_idx = used_df[['poverty_index']].dropna().nsmallest(int(top_n), 'poverty_index').index
                        st.subheader(f"Top {int(top_n)} poorest households (raw data)")
                        st.dataframe(used_df.loc[poorest_idx])
                    else:
                        model_df = used_df[[target_column] + indep_vars].dropna()
                        if model_df.empty:
                            st.error("Not enough data for selected variable combination.")
                        else:
                            X_raw = model_df[indep_vars].copy()
                            y_raw = model_df[target_column].copy()

                            if model_choice == "LogisticRegression":
                                if y_raw.dtype == object or y_raw.dtype.name == 'category' or y_raw.nunique() <= 2:
                                    y = y_raw.astype(str)
                                    uniq = y.unique()
                                    if len(uniq) > 2:
                                        st.error("Dependent variable must have two classes for logistic regression.")
                                        st.stop()
                                    if len(uniq) == 2:
                                        y = (y == uniq[1]).astype(int)
                                    else:
                                        y = y.astype(int)
                                else:
                                    if y_raw.nunique() > 2:
                                        st.error("Dependent variable must have two classes for logistic regression.")
                                        st.stop()
                                    y = y_raw.astype(int)
                            else:
                                y = pd.to_numeric(y_raw, errors='coerce')
                                if y.isna().all():
                                    st.error("Dependent variable is not numeric or could not be converted.")
                                    st.stop()

                            cat_cols_X = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
                            num_cols_X = [c for c in X_raw.columns if c not in cat_cols_X]

                            X_num = X_raw[num_cols_X].apply(pd.to_numeric, errors='coerce')
                            X_cat = pd.get_dummies(X_raw[cat_cols_X].astype(str), drop_first=True) if cat_cols_X else pd.DataFrame(index=X_raw.index)

                            X = pd.concat([X_num, X_cat], axis=1).loc[y.index].dropna()
                            y = y.loc[X.index]

                            if X.empty or y.empty:
                                st.error("Non-numeric or missing data remains, or data lost after categorical encoding.")
                                st.stop()

                            scaler = StandardScaler()
                            Xs = scaler.fit_transform(X)

                            if model_choice == "LinearRegression":
                                lr = LinearRegression().fit(Xs, y)
                                preds = lr.predict(Xs)
                                coefs = pd.Series(lr.coef_, index=X.columns).sort_values(key=abs, ascending=False)
                                st.subheader("Linear Regression - Coefficients")
                                st.dataframe(coefs.rename("coef"))
                                model_df_res = model_df.loc[X.index].copy()
                                model_df_res['predicted'] = preds
                            elif model_choice == "RobustRegression":
                                huber = HuberRegressor().fit(Xs, y)
                                preds = huber.predict(Xs)
                                coefs = pd.Series(huber.coef_, index=X.columns).sort_values(key=abs, ascending=False)
                                st.subheader("Robust Regression (Huber) - Coefficients")
                                st.dataframe(coefs.rename("coef"))
                                model_df_res = model_df.loc[X.index].copy()
                                model_df_res['predicted'] = preds
                            elif model_choice == "RandomForest":
                                rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
                                preds = rf.predict(X)
                                imps = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                                st.subheader("RandomForest - Feature Importances")
                                st.dataframe(imps.rename("importance"))
                                model_df_res = model_df.loc[X.index].copy()
                                model_df_res['predicted'] = preds
                            else:  # LogisticRegression
                                log = LogisticRegression(max_iter=1000).fit(Xs, y)
                                probs = log.predict_proba(Xs)[:, 1]
                                preds = probs
                                st.subheader("Logistic Regression - Coefficients")
                                coefs = pd.Series(log.coef_.flatten(), index=X.columns).sort_values(key=abs, ascending=False)
                                st.dataframe(coefs.rename("coef"))
                                model_df_res = model_df.loc[X.index].copy()
                                model_df_res['predicted_prob'] = probs

                            if model_choice == "LogisticRegression":
                                poorest_idx = model_df_res.nlargest(int(top_n), 'predicted_prob').index
                                st.subheader(f"Top {int(top_n)} households with highest poverty probability (model prediction)")
                                st.dataframe(model_df_res.loc[poorest_idx])
                                st.subheader("All information for these households from the original dataset")
                                st.dataframe(used_df.loc[poorest_idx])
                            else:
                                poorest_idx = model_df_res.nsmallest(int(top_n), 'predicted').index
                                st.subheader(f"Top {int(top_n)} poorest households (model prediction)")
                                st.dataframe(model_df_res.loc[poorest_idx])
                                st.subheader("All information for these households from the original dataset")
                                st.dataframe(used_df.loc[poorest_idx])

                            if model_choice == "LogisticRegression":
                                metric = 'predicted_prob'
                                poorest_group = model_df_res[model_df_res[metric] >= np.percentile(model_df_res[metric], 90)]
                            else:
                                metric = 'predicted'
                                decile_cut = np.percentile(model_df_res[metric], 10)
                                poorest_group = model_df_res[model_df_res[metric] <= decile_cut]

                            profile = poorest_group[indep_vars].apply(pd.to_numeric, errors='coerce').mean().to_frame("poorest_mean")
                            for c in indep_vars:
                                if c in model_df_res.columns and model_df_res[c].dtype == object:
                                    profile.loc[c, 'poorest_mean'] = np.nan
                                    profile.loc[c, 'overall_mean'] = np.nan
                            profile['overall_mean'] = model_df_res[indep_vars].apply(pd.to_numeric, errors='coerce').mean().values
                            st.subheader("Average characteristics of the poorest vs overall mean")
                            st.dataframe(profile)

                            csv = model_df_res.reset_index().to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download prediction results (CSV)", data=csv, file_name="targeting_results.csv", mime="text/csv")

        # ------------------------------
        # 4) SEVERITY
        # ------------------------------
        with tab_severity:
            st.header("Severity - Descriptive by Severity Groups")
            st.markdown("The dataset must already contain variables named 'severity_groups' (categorical) and 'severity_index' (numeric). Select additional categorical breakdown variables to slice the descriptive analysis.")

            # verify required columns exist
            if 'severity_groups' not in df.columns or 'severity_index' not in df.columns:
                st.error("Dataset must contain columns named 'severity_groups' and 'severity_index'.")
            else:
                # allow user to pick categorical breakdown variables (these will be used to further split/group)
                breakdown_choices = [c for c in cat_cols if c != 'severity_groups']
                breakdown_vars = st.multiselect("Categorical breakdown variables (optional)", breakdown_choices, key="severity_breakdowns")

                chart_type = st.selectbox("Chart type", ["Box plot", "Violin plot", "Bar (group mean)", "Histogram", "Density (KDE)"], index=0, key="severity_chart")
                max_levels_warn = 30

                if st.button("Run severity analysis", key="run_severity"):
                    cols = ['severity_groups', 'severity_index'] + breakdown_vars
                    df_sev = df.loc[:, cols].dropna()
                    if df_sev.empty:
                        st.error("No valid data after dropping missing values for the selected variables.")
                    else:
                        # if multiple breakdown vars, create a combined breakdown label
                        if breakdown_vars:
                            df_sev['_breakdown'] = df_sev[breakdown_vars].astype(str).agg(" | ".join, axis=1)
                            breakdown_col = '_breakdown'
                        else:
                            breakdown_col = None

                        # counts of groups
                        n_sev_groups = df_sev['severity_groups'].nunique()
                        if n_sev_groups > max_levels_warn:
                            st.warning(f"'severity_groups' has {n_sev_groups} distinct levels — consider collapsing levels.")

                        if breakdown_col:
                            n_blevels = df_sev[breakdown_col].nunique()
                            if n_blevels > max_levels_warn:
                                st.warning(f"Combined breakdown has {n_blevels} distinct levels — plots may be cluttered.")

                        # Descriptive stats by severity_groups (and breakdown if provided)
                        if breakdown_col:
                            desc = df_sev.groupby(['severity_groups', breakdown_col])['severity_index'].agg(
                                count='count',
                                mean='mean',
                                median=lambda x: x.median(),
                                std='std',
                                min='min',
                                p25=lambda x: np.percentile(x,25),
                                p75=lambda x: np.percentile(x,75),
                                max='max'
                            ).reset_index()
                        else:
                            desc = df_sev.groupby('severity_groups')['severity_index'].agg(
                                count='count',
                                mean='mean',
                                median=lambda x: x.median(),
                                std='std',
                                min='min',
                                p25=lambda x: np.percentile(x,25),
                                p75=lambda x: np.percentile(x,75),
                                max='max'
                            ).reset_index()

                        st.subheader("Group-level descriptive statistics")
                        st.dataframe(desc.round(4))

                        # Show group sizes
                        st.subheader("Group sizes")
                        if breakdown_col:
                            sizes = df_sev.groupby(['severity_groups', breakdown_col]).size().reset_index(name='count')
                        else:
                            sizes = df_sev['severity_groups'].value_counts().rename_axis('severity_groups').reset_index(name='count')
                        st.dataframe(sizes)

                        # Visualization
                        st.subheader("Visualization")
                        try:
                            if chart_type in ("Box plot", "Violin plot"):
                                if breakdown_col:
                                    if chart_type == "Box plot":
                                        fig = px.box(df_sev, x='severity_groups', y='severity_index', color=breakdown_col, points="outliers", title=f"Box plot of severity_index by severity_groups and breakdown")
                                    else:
                                        fig = px.violin(df_sev, x='severity_groups', y='severity_index', color=breakdown_col, box=True, points="outliers", title=f"Violin plot of severity_index by severity_groups and breakdown")
                                else:
                                    if chart_type == "Box plot":
                                        fig = px.box(df_sev, x='severity_groups', y='severity_index', points="outliers", title=f"Box plot of severity_index by severity_groups")
                                    else:
                                        fig = px.violin(df_sev, x='severity_groups', y='severity_index', box=True, points="outliers", title=f"Violin plot of severity_index by severity_groups")
                                st.plotly_chart(fig, use_container_width=True)

                            elif chart_type == "Bar (group mean)":
                                if breakdown_col:
                                    agg = df_sev.groupby(['severity_groups', breakdown_col])['severity_index'].mean().reset_index(name='mean')
                                    fig = px.bar(agg, x='severity_groups', y='mean', color=breakdown_col, barmode='group', title=f"Group mean of severity_index by severity_groups and breakdown")
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.dataframe(agg.round(4))
                                else:
                                    agg = df_sev.groupby('severity_groups')['severity_index'].mean().reset_index(name='mean')
                                    fig = px.bar(agg, x='severity_groups', y='mean', title=f"Group mean of severity_index by severity_groups")
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.dataframe(agg.round(4))

                            elif chart_type == "Histogram":
                                group_sel = st.selectbox("Histogram: choose severity_groups or All", ['All'] + sorted(df_sev['severity_groups'].unique().tolist()), key="hist_group_sel")
                                if group_sel == 'All':
                                    if breakdown_col:
                                        fig = px.histogram(df_sev, x='severity_index', color=breakdown_col, nbins=40, title="Histogram of severity_index (all severity groups, colored by breakdown)", marginal="rug")
                                    else:
                                        fig = px.histogram(df_sev, x='severity_index', nbins=40, title="Histogram of severity_index (all groups)", marginal="rug")
                                else:
                                    subset = df_sev[df_sev['severity_groups'] == group_sel]
                                    if breakdown_col:
                                        fig = px.histogram(subset, x='severity_index', color=breakdown_col, nbins=40, title=f"Histogram of severity_index for {group_sel} (colored by breakdown)", marginal="rug")
                                    else:
                                        fig = px.histogram(subset, x='severity_index', nbins=40, title=f"Histogram of severity_index for {group_sel}", marginal="rug")
                                st.plotly_chart(fig, use_container_width=True)

                            else:  # Density (approx via histograms normalized)
                                sel_groups = st.multiselect("Density: select breakdown levels to plot (leave empty = all)", sorted(df_sev[breakdown_col].unique().tolist()) if breakdown_col else [], default=None, key="density_groups")
                                if breakdown_col:
                                    plot_df = df_sev if not sel_groups else df_sev[df_sev[breakdown_col].isin(sel_groups)]
                                    if plot_df.empty:
                                        st.warning("No data for selected breakdown levels.")
                                    else:
                                        fig = px.histogram(plot_df, x='severity_index', color=breakdown_col, histnorm='density', nbins=60, barmode='overlay', opacity=0.6, title="Density (approx.) of severity_index by breakdown")
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig = px.histogram(df_sev, x='severity_index', histnorm='density', nbins=60, title="Density (approx.) of severity_index")
                                    st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Plotting failed: {e}")

                        # Offer downloads
                        csv_desc = desc.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download group descriptive stats (CSV)", data=csv_desc, file_name="severity_groups_stats.csv", mime="text/csv")
                        csv_raw = df_sev.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download filtered raw data (CSV)", data=csv_raw, file_name="severity_filtered_data.csv", mime="text/csv")