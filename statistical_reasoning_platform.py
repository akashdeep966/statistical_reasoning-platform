
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import json
import pickle
import zipfile
import warnings
warnings.filterwarnings('ignore')

# --- Configuration & Styling ---
st.set_page_config(page_title="Statistical Reasoning & Analytics Platform", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #3B82F6; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; }
    .info-box { background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .warning-box { background-color: #FFFBEB; border-left: 5px solid #F59E0B; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .success-box { background-color: #ECFDF5; border-left: 5px solid #10B981; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .danger-box { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Semantic Relevance Engine ---
SEMANTIC_KEYWORDS = {
    'physical_size': ['height', 'length', 'width', 'weight', 'size', 'volume', 'area'],
    'financial': ['salary', 'income', 'revenue', 'profit', 'cost', 'price', 'expense', 'budget', 'gdp', 'inflation'],
    'temporal': ['date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'second', 'age', 'duration'],
    'demographic': ['gender', 'sex', 'race', 'ethnicity', 'nationality', 'religion', 'marital'],
    'location': ['city', 'country', 'state', 'region', 'zip', 'postal', 'latitude', 'longitude', 'address'],
    'identifier': ['id', 'code', 'number', 'uuid', 'key', 'index'],
    'performance': ['score', 'grade', 'rating', 'rank', 'performance', 'efficiency', 'accuracy']
}

def get_semantic_category(var_name):
    var_lower = var_name.lower().replace('_', ' ')
    for category, keywords in SEMANTIC_KEYWORDS.items():
        if any(kw in var_lower for kw in keywords):
            return category
    return 'general'

def calculate_semantic_relevance(var1, var2):
    cat1 = get_semantic_category(var1)
    cat2 = get_semantic_category(var2)
    if cat1 == cat2:
        return 85
    logical_pairs = [
        ('financial', 'performance'), ('physical_size', 'performance'), 
        ('temporal', 'financial'), ('demographic', 'financial'),
        ('location', 'financial'), ('temporal', 'performance')
    ]
    if (cat1, cat2) in logical_pairs or (cat2, cat1) in logical_pairs:
        return 70
    unrelated_pairs = [
        ('physical_size', 'identifier'), ('location', 'identifier'),
        ('demographic', 'identifier'), ('temporal', 'identifier')
    ]
    if (cat1, cat2) in unrelated_pairs or (cat2, cat1) in unrelated_pairs:
        return 10
    return 50

def check_normality(series):
    if len(series) > 5000:
        stat, p_value = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
        test_name = "Kolmogorov-Smirnov"
    else:
        stat, p_value = stats.shapiro(series)
        test_name = "Shapiro-Wilk"
    is_normal = p_value > 0.05
    interpretation = f"{test_name}: p={p_value:.4f}. Data is {'Normally Distributed' if is_normal else 'Not Normally Distributed'}."
    return {"test": test_name, "statistic": stat, "p_value": p_value, "is_normal": is_normal, "interpretation": interpretation}

def check_multicollinearity(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif = vif_data[vif_data['VIF'] > 10]
    interpretation = f"Multicollinearity check: {len(high_vif)} features with VIF > 10."
    return {"vif_table": vif_data, "high_vif_features": high_vif['feature'].tolist(), "interpretation": interpretation}

def get_download_link(object_to_download, download_filename, link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
    else:
        b64 = base64.b64encode(object_to_download).decode()
    return f'<a href="data:file/zip;base64,{b64}" download="{download_filename}">{link_text}</a>'

def create_zip_package(model, scaler, encoders, feature_names, target_name, metrics):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('model.pkl', pickle.dumps(model))
        if scaler:
            zip_file.writestr('scaler.pkl', pickle.dumps(scaler))
        if encoders:
            zip_file.writestr('encoders.pkl', pickle.dumps(encoders))
        metadata = {
            'feature_names': feature_names,
            'target_name': target_name,
            'metrics': metrics,
            'creation_date': str(pd.Timestamp.now()),
            'version': '1.0.0'
        }
        zip_file.writestr('metadata.json', json.dumps(metadata, indent=4))
    return zip_buffer.getvalue()

def main():
    st.markdown('<div class="main-header">🧠 Statistical Reasoning & Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("*A system that thinks before it calculates.*")

    with st.sidebar:
        st.header("⚙️ Configuration")
        analysis_objective = st.selectbox(
            "Analysis Objective",
            ["Prediction", "Explanation", "Hypothesis Testing", "Segmentation", "Dimensional Reduction", "Time Series Forecasting"]
        )
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        st.markdown("1. Upload Data")
        st.markdown("2. Define Variables")
        st.markdown("3. Data Quality")
        st.markdown("4. EDA & Assumptions")
        st.markdown("5. Method Selection")
        st.markdown("6. Model Training")
        st.markdown("7. Interpretation & Deployment")

    # --- Stage 1: Data Upload ---
    st.markdown('<div class="sub-header">Stage 1: Dataset Upload & Understanding</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Dataset (CSV, Excel, TSV)", type=['csv', 'xlsx', 'tsv'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
            st.session_state['df'] = df
            st.session_state['filename'] = uploaded_file.name
            st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
    else:
        st.info("No file uploaded. Using built-in **Diabetes Dataset** for demonstration.")
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.session_state['df'] = df
        st.session_state['filename'] = 'diabetes_demo.csv'
        st.write(df.head())

    df = st.session_state['df']

    # --- Stage 2: Semantic Variable Understanding ---
    st.markdown('<div class="sub-header">Stage 2: Semantic Variable Understanding</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    The system automatically classifies variables into semantic categories (e.g., Financial, Physical, Temporal) 
    to guard against spurious correlations and ensure domain-relevant analysis.
    </div>
    """, unsafe_allow_html=True)

    var_metadata = []
    for col in df.columns:
        var_metadata.append({
            'Variable': col,
            'Type': str(df[col].dtype),
            'Semantic Category': get_semantic_category(col),
            'Unique Values': df[col].nunique(),
            'Missing (%)': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%"
        })

    meta_df = pd.DataFrame(var_metadata)
    st.dataframe(meta_df, use_container_width=True)

    if len(df.columns) <= 15:
        relevance_matrix = np.zeros((len(df.columns), len(df.columns)))
        for i, c1 in enumerate(df.columns):
            for j, c2 in enumerate(df.columns):
                if i != j:
                    relevance_matrix[i, j] = calculate_semantic_relevance(c1, c2)

        fig = px.imshow(relevance_matrix, x=df.columns, y=df.columns, 
                        color_continuous_scale='RdYlGn', zmin=0, zmax=100,
                        title="Semantic Relevance Score (0-100)")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # --- Stage 3: Data Quality Assessment ---
    st.markdown('<div class="sub-header">Stage 3: Data Quality Assessment</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col2:
        st.metric("Duplicate Rows", df.duplicated().sum())
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        st.warning(f"Missing values detected in: {', '.join(missing_cols)}")
        missing_df = df[missing_cols].isnull().sum().reset_index()
        missing_df.columns = ['Variable', 'Missing Count']
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
        fig = px.bar(missing_df, x='Variable', y='Missing %', color='Missing %', 
                     color_continuous_scale='Reds', title="Missing Value Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        <b>Recommendation:</b> For numerical variables, use <b>Median Imputation</b> (robust to outliers). 
        For categorical, use <b>Mode Imputation</b>. If missingness > 30%, consider <b>Multiple Imputation</b> or deletion.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("No missing values detected. Data is complete.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_outlier_col = st.selectbox("Select variable for Outlier Analysis", numeric_cols)
        Q1 = df[selected_outlier_col].quantile(0.25)
        Q3 = df[selected_outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[selected_outlier_col] < (Q1 - 1.5 * IQR)) | (df[selected_outlier_col] > (Q3 + 1.5 * IQR))]

        fig = go.Figure()
        fig.add_trace(go.Box(y=df[selected_outlier_col], name=selected_outlier_col, boxmean='sd'))
        fig.update_layout(title=f"Box Plot for {selected_outlier_col} (Outliers: {len(outliers)})")
        st.plotly_chart(fig, use_container_width=True)

        if len(outliers) > 0:
            st.markdown(f"""
            <div class="warning-box">
            <b>Outlier Analysis:</b> {len(outliers)} outliers detected using IQR method. 
            These could be <b>Data Errors</b> or <b>Legitimate Extreme Observations</b>. 
            Do not remove without domain validation.
            </div>
            """, unsafe_allow_html=True)

    # --- Stage 4: EDA ---
    st.markdown('<div class="sub-header">Stage 4: Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Univariate", "Bivariate", "Multivariate"])

    with eda_tab1:
        if numeric_cols:
            selected_dist_col = st.selectbox("Select variable for Distribution Analysis", numeric_cols, key="dist")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_dist_col, marginal="box", nbins=30, 
                                 title=f"Distribution of {selected_dist_col}",
                                 color_discrete_sequence=['#3B82F6'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                skew = df[selected_dist_col].skew()
                kurt = df[selected_dist_col].kurtosis()
                st.markdown(f"**Skewness:** {skew:.2f} ({'Right' if skew > 0 else 'Left'} skewed)")
                st.markdown(f"**Kurtosis:** {kurt:.2f} ({'Heavy' if kurt > 3 else 'Light'} tails)")
                if abs(skew) > 1:
                    st.warning("High skewness detected. Consider log transformation for parametric modeling.")
                else:
                    st.success("Distribution is relatively symmetric.")

    with eda_tab2:
        if len(numeric_cols) >= 2:
            corr_method = st.radio("Correlation Method", ["Pearson (Linear)", "Spearman (Monotonic)"], horizontal=True)
            method = 'pearson' if 'Pearson' in corr_method else 'spearman'
            corr_matrix = df[numeric_cols].corr(method=method)

            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                            title=f"{method.capitalize()} Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        var1 = corr_matrix.columns[i]
                        var2 = corr_matrix.columns[j]
                        relevance = calculate_semantic_relevance(var1, var2)
                        if relevance < 30:
                            high_corr_pairs.append(f"⚠️ **{var1}** vs **{var2}**: r={corr_matrix.iloc[i, j]:.2f} (Semantic Relevance: {relevance}/100). **Potential Spurious Correlation!**")

            if high_corr_pairs:
                st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                st.markdown("### Semantic Guardrail Alerts")
                for alert in high_corr_pairs:
                    st.markdown(alert)
                st.markdown('</div>', unsafe_allow_html=True)

    with eda_tab3:
        if len(numeric_cols) >= 3:
            pca = PCA(n_components=3)
            pca_components = pca.fit_transform(df[numeric_cols].dropna())
            pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2', 'PC3'])
            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', 
                              title=f"3D PCA Projection (Explained Var: {sum(pca.explained_variance_ratio_):.2%})")
            st.plotly_chart(fig, use_container_width=True)

    # --- Stage 5: Assumption Engine ---
    st.markdown('<div class="sub-header">Stage 5: Assumption Engine</div>', unsafe_allow_html=True)

    if st.button("Run Full Assumption Diagnostics", type="primary"):
        with st.spinner("Running statistical tests..."):
            st.markdown("#### 1. Normality Tests")
            normality_results = []
            for col in numeric_cols[:5]:
                res = check_normality(df[col].dropna())
                normality_results.append({"Variable": col, **res})
            norm_df = pd.DataFrame(normality_results)
            st.dataframe(norm_df[['Variable', 'test', 'p_value', 'interpretation']], use_container_width=True)

            target_col = st.session_state.get('target_col')
            if target_col and target_col in numeric_cols and len(numeric_cols) > 1:
                st.markdown("#### 2. Homoscedasticity Test")
                X = df[numeric_cols].drop(columns=[target_col]).dropna()
                y = df[target_col].loc[X.index]
                if not X.empty:
                    X_const = sm.add_constant(X)
                    model = sm.OLS(y, X_const).fit()
                    bp_test = het_breuschpagan(model.resid, model.model.exog)
                    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
                    result = dict(zip(labels, bp_test))
                    is_homoscedastic = result['LM-Test p-value'] > 0.05
                    st.info(f"Breusch-Pagan: p={result['LM-Test p-value']:.4f}. {'Homoscedastic' if is_homoscedastic else 'Heteroscedastic'} errors detected.")

            st.markdown("#### 3. Multicollinearity (VIF)")
            if len(numeric_cols) > 1:
                X_vif = df[numeric_cols].dropna()
                vif_res = check_multicollinearity(X_vif)
                st.dataframe(vif_res['vif_table'], use_container_width=True)
                if vif_res['high_vif_features']:
                    st.warning(f"High multicollinearity detected in: {', '.join(vif_res['high_vif_features'])}")

    # --- Stage 6: Method Selection & Model Training ---
    st.markdown('<div class="sub-header">Stage 6: Statistical Method Selection & Model Training</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    Based on your objective (<b>{analysis_objective}</b>), the system recommends the following pipeline. 
    Assumptions checked above dictate whether Parametric or Non-parametric methods are used.
    </div>
    """, unsafe_allow_html=True)

    target_col = st.selectbox("Select Target Variable (if applicable)", ['None'] + df.columns.tolist())
    if target_col != 'None':
        st.session_state['target_col'] = target_col

    if analysis_objective == "Prediction" and target_col != 'None':
        is_classification = df[target_col].dtype == 'object' or df[target_col].nunique() < 10

        st.markdown(f"### Objective: Predict **{target_col}**")
        st.markdown(f"**Detected Task Type:** {'Classification' if is_classification else 'Regression'}")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        numeric_X_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        imputer = SimpleImputer(strategy='median')
        X[numeric_X_cols] = imputer.fit_transform(X[numeric_X_cols])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if is_classification:
            le_y = LabelEncoder()
            y_train = le_y.fit_transform(y_train)
            y_test = le_y.transform(y_test)
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }
            metrics_func = {
                'Accuracy': accuracy_score, 'F1': lambda y, p: f1_score(y, p, average='weighted'), 
                'Precision': lambda y, p: precision_score(y, p, average='weighted'), 
                'Recall': lambda y, p: recall_score(y, p, average='weighted')
            }
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Elastic Net': ElasticNet(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            metrics_func = {
                'RMSE': lambda y, p: np.sqrt(mean_squared_error(y, p)),
                'MAE': mean_absolute_error, 'R2': r2_score
            }

        if st.button("Train & Compare Models", type="primary"):
            with st.spinner("Training models..."):
                results = []
                trained_models = {}

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    trained_models[name] = model

                    row = {'Model': name}
                    for metric_name, func in metrics_func.items():
                        try:
                            row[metric_name] = func(y_test, y_pred)
                        except:
                            row[metric_name] = np.nan
                    results.append(row)

                results_df = pd.DataFrame(results)
                st.session_state['results_df'] = results_df
                st.session_state['trained_models'] = trained_models
                st.session_state['scaler'] = scaler
                st.session_state['encoders'] = encoders
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['is_classification'] = is_classification
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test

                st.markdown("### Model Performance Comparison")
                st.dataframe(results_df, use_container_width=True)

                fig = px.bar(results_df, x='Model', y=results_df.columns[1], 
                           color='Model', title=f"Model Comparison: {results_df.columns[1]}")
                st.plotly_chart(fig, use_container_width=True)

                best_idx = results_df[results_df.columns[1]].idxmax() if is_classification else results_df[results_df.columns[1]].idxmin()
                best_model_name = results_df.loc[best_idx, 'Model']
                st.success(f"🏆 Best Performing Model: **{best_model_name}**")
                st.session_state['best_model'] = trained_models[best_model_name]
                st.session_state['best_model_name'] = best_model_name

    # --- Stage 7: Explainability & Deployment ---
    if 'best_model' in st.session_state:
        st.markdown('<div class="sub-header">Stage 7: Interpretation & Deployment</div>', unsafe_allow_html=True)

        best_model = st.session_state['best_model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        st.markdown("### Feature Importance (Permutation)")
        perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'Feature': st.session_state['feature_names'],
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                     error_x='Std', title="Permutation Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>AI Interpretation:</b> The chart above shows which variables most significantly impact the model's predictions. 
        High importance suggests a strong predictive relationship, but remember: <b>Correlation does not imply Causation</b>. 
        Validate these relationships with domain knowledge.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Practical Significance Assessment")
        st.markdown("""
        <div class="warning-box">
        Even if a model achieves high statistical performance (e.g., R² > 0.9), always ask: 
        <b>"Does a 1% improvement in accuracy justify the complexity of this model in a real-world business scenario?"</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Model Registry & Deployment Package")
        metrics = st.session_state['results_df'].to_dict('records')
        zip_data = create_zip_package(
            best_model, st.session_state['scaler'], st.session_state['encoders'],
            st.session_state['feature_names'], target_col, metrics
        )

        st.markdown(get_download_link(zip_data, "analytical_pipeline_v1.0.zip", "📦 Download Deployable Pipeline (ZIP)"), unsafe_allow_html=True)
        st.markdown("""
        <div class="success-box">
        This package contains the model, scaler, encoders, and metadata. 
        You can load this in a production environment using the provided <code>metadata.json</code> to ensure feature consistency.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<center>Powered by Statistical Reasoning Engine v1.0 | Built for Analysts, Not Just Machines</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
