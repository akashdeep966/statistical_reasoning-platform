
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, silhouette_score, mean_absolute_percentage_error)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import json
import pickle
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and SHAP
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- Configuration ---
st.set_page_config(page_title="Statistical Reasoning & Analytics Platform", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #3B82F6; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; }
    .info-box { background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .warning-box { background-color: #FFFBEB; border-left: 5px solid #F59E0B; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .success-box { background-color: #ECFDF5; border-left: 5px solid #10B981; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .danger-box { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .insight-box { background-color: #F3E8FF; border-left: 5px solid #9333EA; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .metric-card { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 0.5rem; padding: 1rem; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #1E3A8A; }
    .metric-label { font-size: 0.875rem; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# --- Semantic Engine ---
SEMANTIC_KEYWORDS = {
    'physical_size': ['height', 'length', 'width', 'weight', 'size', 'volume', 'area', 'mass', 'depth'],
    'financial': ['salary', 'income', 'revenue', 'profit', 'cost', 'price', 'expense', 'budget', 'gdp', 'inflation', 'wage', 'earnings', 'sales', 'revenue'],
    'temporal': ['date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'second', 'age', 'duration', 'period', 'tenure'],
    'demographic': ['gender', 'sex', 'race', 'ethnicity', 'nationality', 'religion', 'marital', 'education'],
    'location': ['city', 'country', 'state', 'region', 'zip', 'postal', 'latitude', 'longitude', 'address', 'location'],
    'identifier': ['id', 'code', 'number', 'uuid', 'key', 'index', 'serial'],
    'performance': ['score', 'grade', 'rating', 'rank', 'performance', 'efficiency', 'accuracy', 'precision', 'recall'],
    'medical': ['bp', 'blood', 'glucose', 'cholesterol', 'bmi', 'heart', 'disease', 'diagnosis', 'symptom', 'patient'],
    'behavioral': ['click', 'view', 'purchase', 'visit', 'engagement', 'conversion', 'bounce', 'session']
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
        ('location', 'financial'), ('temporal', 'performance'),
        ('medical', 'physical_size'), ('medical', 'performance'),
        ('behavioral', 'performance'), ('behavioral', 'financial')
    ]
    if (cat1, cat2) in logical_pairs or (cat2, cat1) in logical_pairs:
        return 70
    unrelated_pairs = [
        ('physical_size', 'identifier'), ('location', 'identifier'),
        ('demographic', 'identifier'), ('temporal', 'identifier'),
        ('medical', 'identifier'), ('financial', 'identifier')
    ]
    if (cat1, cat2) in unrelated_pairs or (cat2, cat1) in unrelated_pairs:
        return 10
    return 50

def get_semantic_reasoning(var1, var2, relevance):
    """Generate human-readable reasoning about variable relationships."""
    cat1 = get_semantic_category(var1)
    cat2 = get_semantic_category(var2)
    if relevance >= 80:
        return f"Both variables are in the same semantic category ({cat1}), suggesting a plausible domain relationship."
    elif relevance >= 60:
        return f"Variables belong to logically related categories ({cat1} and {cat2}), suggesting potential meaningful association."
    elif relevance <= 20:
        return f"⚠️ Variables belong to unrelated categories ({cat1} and {cat2}). Any correlation may be spurious or confounded."
    else:
        return f"Variables have moderate semantic distance. Relationship requires domain validation."

# --- Statistical Tests ---
def check_normality(series):
    if len(series) > 5000:
        stat, p_value = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
        test_name = "Kolmogorov-Smirnov"
    else:
        stat, p_value = stats.shapiro(series)
        test_name = "Shapiro-Wilk"
    is_normal = p_value > 0.05
    interpretation = f"{test_name}: p={'%.4f' % p_value}. Data is {'Normally Distributed' if is_normal else 'Not Normally Distributed'}."
    return {"test": test_name, "statistic": stat, "p_value": p_value, "is_normal": is_normal, "interpretation": interpretation}

def check_multicollinearity(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif = vif_data[vif_data['VIF'] > 10]
    interpretation = f"Multicollinearity check: {len(high_vif)} features with VIF > 10."
    return {"vif_table": vif_data, "high_vif_features": high_vif['feature'].tolist(), "interpretation": interpretation}

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std

def interpret_effect_size(d):
    """Interpret Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

def get_download_link(object_to_download, download_filename, link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
    else:
        b64 = base64.b64encode(object_to_download).decode()
    return f'<a href="data:file/zip;base64,{b64}" download="{download_filename}">{link_text}</a>'

def create_zip_package(model, scaler, encoders, feature_names, target_name, metrics, assumptions):
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
            'assumptions': assumptions,
            'creation_date': str(pd.Timestamp.now()),
            'version': '2.0.0'
        }
        zip_file.writestr('metadata.json', json.dumps(metadata, indent=4))
    return zip_buffer.getvalue()

# --- Model Selection Engine ---
def select_models(task_type, assumptions):
    """Select appropriate models based on assumption test results."""
    models = {}
    reasoning = []

    if task_type == 'regression':
        # Always include linear models for interpretability baseline
        models['Linear Regression'] = LinearRegression()
        reasoning.append("Linear Regression: Baseline interpretable model.")

        if assumptions.get('is_normal', False):
            models['Ridge'] = Ridge(alpha=1.0)
            models['Lasso'] = Lasso(alpha=0.1)
            models['Elastic Net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
            reasoning.append("Ridge/Lasso/ElasticNet: Data is approximately normal, regularized linear models are appropriate.")
        else:
            reasoning.append("⚠️ Data is non-normal. Linear models may be biased. Tree-based models recommended.")

        # Tree-based models (robust to non-normality and outliers)
        models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reasoning.append("Random Forest: Robust to non-normality and outliers. Good for capturing non-linear relationships.")

        models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reasoning.append("Gradient Boosting: Sequential ensemble, often higher accuracy than Random Forest.")

        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            reasoning.append("XGBoost: State-of-the-art gradient boosting with regularization.")

        # Non-parametric alternative
        models['KNN'] = KNeighborsRegressor(n_neighbors=5)
        reasoning.append("KNN: Non-parametric, useful for local patterns. Sensitive to feature scaling.")

    else:  # classification
        models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        reasoning.append("Logistic Regression: Baseline probabilistic classifier with interpretable coefficients.")

        models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        reasoning.append("Random Forest: Robust ensemble, handles class imbalance well.")

        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        reasoning.append("Gradient Boosting: Often highest accuracy for tabular data.")

        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
            reasoning.append("XGBoost: Optimized gradient boosting with built-in regularization.")

        models['KNN'] = KNeighborsClassifier(n_neighbors=5)
        reasoning.append("KNN: Instance-based classifier. Good for small datasets with clear decision boundaries.")

    return models, reasoning

def evaluate_model_practical_significance(y_true, y_pred, task_type, X_test=None, feature_names=None):
    """Calculate practical significance metrics beyond statistical performance."""
    results = {}

    if task_type == 'regression':
        # Standard metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        results['RMSE'] = rmse
        results['MAE'] = mae
        results['R2'] = r2
        results['MAPE'] = mape

        # Practical significance: what does MAPE mean in business terms?
        if mape < 5:
            results['practical_rating'] = "Excellent"
            results['practical_message'] = f"Predictions are off by only {mape:.1f}% on average. Highly reliable for decision-making."
        elif mape < 10:
            results['practical_rating'] = "Good"
            results['practical_message'] = f"Predictions are off by {mape:.1f}% on average. Useful for planning, but verify critical decisions."
        elif mape < 20:
            results['practical_rating'] = "Moderate"
            results['practical_message'] = f"Predictions are off by {mape:.1f}% on average. Use as directional guidance only."
        else:
            results['practical_rating'] = "Poor"
            results['practical_message'] = f"Predictions are off by {mape:.1f}% on average. Model needs significant improvement before deployment."

        # Effect size: how much variance is explained?
        if r2 > 0.7:
            results['variance_message'] = f"R² = {r2:.3f}: Model explains {r2*100:.1f}% of variance. Strong predictive power."
        elif r2 > 0.4:
            results['variance_message'] = f"R² = {r2:.3f}: Model explains {r2*100:.1f}% of variance. Moderate predictive power. Other factors likely influence the outcome."
        else:
            results['variance_message'] = f"R² = {r2:.3f}: Model explains only {r2*100:.1f}% of variance. Weak predictive power. Consider additional features or non-linear transformations."

    else:  # classification
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Check if accuracy is better than baseline (majority class)
        majority_baseline = pd.Series(y_true).value_counts(normalize=True).iloc[0]
        lift = (acc - majority_baseline) / majority_baseline * 100

        results['Accuracy'] = acc
        results['F1'] = f1
        results['Majority_Baseline'] = majority_baseline
        results['Lift_over_Baseline'] = lift

        if lift > 50:
            results['practical_rating'] = "Excellent"
            results['practical_message'] = f"Model achieves {acc:.1%} accuracy, {lift:.1f}% better than always predicting the majority class. Strong business value."
        elif lift > 20:
            results['practical_rating'] = "Good"
            results['practical_message'] = f"Model achieves {acc:.1f}% accuracy, {lift:.1f}% better than baseline. Provides meaningful improvement."
        elif lift > 0:
            results['practical_rating'] = "Marginal"
            results['practical_message'] = f"Model achieves {acc:.1f}% accuracy, only {lift:.1f}% better than baseline. Limited practical value."
        else:
            results['practical_rating'] = "Poor"
            results['practical_message'] = f"Model performs worse than simply predicting the majority class. Do not deploy."

    return results

# --- Main Application ---
def main():
    st.markdown('<div class="main-header">🧠 Statistical Reasoning & Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("*A system that thinks before it calculates. Statistical significance ≠ Practical significance.*")

    with st.sidebar:
        st.header("⚙️ Configuration")
        analysis_objective = st.selectbox(
            "Analysis Objective",
            ["Prediction", "Explanation", "Hypothesis Testing", "Segmentation", "Dimensional Reduction"]
        )
        st.markdown("---")
        st.markdown("### 🧭 Pipeline Stages")
        st.markdown("1. Upload & Understand Data")
        st.markdown("2. Semantic Variable Analysis")
        st.markdown("3. Data Quality Assessment")
        st.markdown("4. Exploratory Data Analysis")
        st.markdown("5. Assumption Engine")
        st.markdown("6. Method Selection & Training")
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
            st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
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

    # --- Stage 2: Semantic Understanding ---
    st.markdown('<div class="sub-header">Stage 2: Semantic Variable Understanding</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    The system classifies variables into semantic categories and guards against spurious correlations 
    by evaluating domain plausibility before any statistical computation.
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

    # --- Stage 3: Data Quality ---
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
        For categorical, use <b>Mode Imputation</b>. If missingness > 30%, consider deletion or multiple imputation.
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
        fig.update_layout(title=f"Box Plot: {selected_outlier_col} (Outliers: {len(outliers)})")
        st.plotly_chart(fig, use_container_width=True)

        if len(outliers) > 0:
            pct_outliers = len(outliers) / len(df) * 100
            st.markdown(f"""
            <div class="warning-box">
            <b>Outlier Analysis:</b> {len(outliers)} outliers ({pct_outliers:.1f}%) detected via IQR. 
            These could be <b>Data Errors</b>, <b>Rare Events</b>, or <b>Legitimate Extremes</b>. 
            Do not remove without domain validation. Consider RobustScaler if outliers are legitimate.
            </div>
            """, unsafe_allow_html=True)

    # --- Stage 4: EDA ---
    st.markdown('<div class="sub-header">Stage 4: Exploratory Data Analysis</div>', unsafe_allow_html=True)

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
                    st.warning("High skewness. Consider log transformation or non-parametric methods.")
                else:
                    st.success("Distribution is relatively symmetric. Parametric methods are appropriate.")

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
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        var1 = corr_matrix.columns[i]
                        var2 = corr_matrix.columns[j]
                        relevance = calculate_semantic_relevance(var1, var2)
                        reasoning = get_semantic_reasoning(var1, var2, relevance)
                        if relevance < 30:
                            high_corr_pairs.append(f"🚩 **{var1}** vs **{var2}**: r={corr_matrix.iloc[i, j]:.2f}\n   {reasoning}")
                        else:
                            high_corr_pairs.append(f"✅ **{var1}** vs **{var2}**: r={corr_matrix.iloc[i, j]:.2f}\n   {reasoning}")

            if high_corr_pairs:
                st.markdown("### Semantic Guardrail Analysis")
                for alert in high_corr_pairs:
                    if "🚩" in alert:
                        st.markdown(f'<div class="danger-box">{alert}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-box">{alert}</div>', unsafe_allow_html=True)

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

    target_col = st.selectbox("Select Target Variable (if applicable)", ['None'] + df.columns.tolist())
    if target_col != 'None':
        st.session_state['target_col'] = target_col

    if st.button("🔬 Run Full Assumption Diagnostics", type="primary"):
        with st.spinner("Running comprehensive statistical tests..."):
            assumption_results = {}

            # Normality
            st.markdown("#### 1. Normality Tests")
            normality_results = []
            for col in numeric_cols[:6]:
                res = check_normality(df[col].dropna())
                normality_results.append({"Variable": col, **res})
            norm_df = pd.DataFrame(normality_results)
            st.dataframe(norm_df[['Variable', 'test', 'p_value', 'interpretation']], use_container_width=True)

            all_normal = all(norm_df['is_normal'])
            assumption_results['is_normal'] = all_normal

            if all_normal:
                st.success("✅ All tested variables are normally distributed. Parametric methods are appropriate.")
            else:
                st.warning("⚠️ Some variables are non-normal. Non-parametric methods or transformations recommended.")

            # Homoscedasticity
            if target_col != 'None' and target_col in numeric_cols and len(numeric_cols) > 1:
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
                    assumption_results['is_homoscedastic'] = is_homoscedastic

                    if is_homoscedastic:
                        st.success(f"✅ Breusch-Pagan: p={result['LM-Test p-value']:.4f}. Homoscedastic errors confirmed.")
                    else:
                        st.warning(f"⚠️ Breusch-Pagan: p={result['LM-Test p-value']:.4f}. Heteroscedastic errors detected. Consider robust standard errors or weighted least squares.")

            # Multicollinearity
            st.markdown("#### 3. Multicollinearity (VIF)")
            if len(numeric_cols) > 1:
                X_vif = df[numeric_cols].dropna()
                vif_res = check_multicollinearity(X_vif)
                st.dataframe(vif_res['vif_table'], use_container_width=True)
                assumption_results['high_vif'] = vif_res['high_vif_features']

                if vif_res['high_vif_features']:
                    st.warning(f"⚠️ High multicollinearity in: {', '.join(vif_res['high_vif_features'])}. Consider removing or combining these features.")
                else:
                    st.success("✅ No severe multicollinearity detected (all VIF < 10).")

            st.session_state['assumption_results'] = assumption_results

    # --- Stage 6: Method Selection & Model Training ---
    st.markdown('<div class="sub-header">Stage 6: Statistical Method Selection & Model Training</div>', unsafe_allow_html=True)

    if target_col != 'None':
        is_classification = df[target_col].dtype == 'object' or df[target_col].nunique() < 10
        task_type = 'classification' if is_classification else 'regression'

        st.markdown(f"### Objective: Predict **{target_col}**")
        st.markdown(f"**Detected Task Type:** `{task_type.upper()}`")

        # Assumption-aware model selection
        assumptions = st.session_state.get('assumption_results', {})
        if assumptions:
            st.markdown("""
            <div class="info-box">
            <b>Assumption-Aware Selection:</b> Models are chosen based on the assumption tests above.
            </div>
            """, unsafe_allow_html=True)

        models, model_reasoning = select_models(task_type, assumptions)

        with st.expander("📋 Why these models were selected"):
            for reason in model_reasoning:
                st.markdown(f"- {reason}")

        # Prepare data
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

        # Use RobustScaler if outliers detected, otherwise StandardScaler
        has_outliers = any(abs(df[c].skew()) > 1.5 for c in numeric_X_cols if c != target_col)
        if has_outliers:
            scaler = RobustScaler()
            st.info("Using RobustScaler due to detected skewness/outliers.")
        else:
            scaler = StandardScaler()
            st.info("Using StandardScaler (data appears well-behaved).")

        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y if is_classification else None)

        if is_classification:
            le_y = LabelEncoder()
            y_train = le_y.fit_transform(y_train)
            y_test = le_y.transform(y_test)

        if st.button("🚀 Train & Compare All Models", type="primary"):
            with st.spinner("Training and cross-validating models..."):
                results = []
                trained_models = {}
                cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if is_classification else KFold(n_splits=5, shuffle=True, random_state=42)

                for name, model in models.items():
                    try:
                        # Cross-validation
                        if is_classification:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted')
                            cv_metric = 'F1'
                        else:
                            cv_scores = -cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_root_mean_squared_error')
                            cv_metric = 'RMSE'

                        # Fit on full training set
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        trained_models[name] = model

                        row = {'Model': name, f'CV {cv_metric} (mean)': cv_scores.mean(), f'CV {cv_metric} (std)': cv_scores.std()}

                        # Test set metrics
                        practical = evaluate_model_practical_significance(y_test, y_pred, task_type)
                        row.update(practical)
                        results.append(row)
                    except Exception as e:
                        st.error(f"Error training {name}: {e}")

                results_df = pd.DataFrame(results)
                st.session_state['results_df'] = results_df
                st.session_state['trained_models'] = trained_models
                st.session_state['scaler'] = scaler
                st.session_state['encoders'] = encoders
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['is_classification'] = is_classification
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['task_type'] = task_type

                # Display results
                st.markdown("### Model Performance Comparison")
                st.dataframe(results_df, use_container_width=True)

                # Best model selection
                if is_classification:
                    best_idx = results_df['Accuracy'].idxmax()
                else:
                    best_idx = results_df['RMSE'].idxmin()

                best_model_name = results_df.loc[best_idx, 'Model']
                st.session_state['best_model'] = trained_models[best_model_name]
                st.session_state['best_model_name'] = best_model_name

                st.success(f"🏆 Best Model: **{best_model_name}**")

                # Practical significance highlight
                best_practical = evaluate_model_practical_significance(
                    y_test, 
                    trained_models[best_model_name].predict(X_test), 
                    task_type
                )

                st.markdown(f"""
                <div class="insight-box">
                <b>Practical Significance Assessment:</b><br>
                {best_practical['practical_message']}<br><br>
                {best_practical.get('variance_message', '')}
                </div>
                """, unsafe_allow_html=True)

    # --- Stage 7: Interpretation & Explainability ---
    if 'best_model' in st.session_state:
        st.markdown('<div class="sub-header">Stage 7: Model Interpretation & Explainability</div>', unsafe_allow_html=True)

        best_model = st.session_state['best_model']
        best_name = st.session_state['best_model_name']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        feature_names = st.session_state['feature_names']
        task_type = st.session_state['task_type']

        st.markdown(f"### Interpreting: **{best_name}**")

        # CAUSAL WARNING - Always show this
        st.markdown("""
        <div class="danger-box">
        <b>⚠️ CAUSAL INFERENCE WARNING:</b><br>
        The following analysis shows <b>predictive relationships</b>, not causal effects. 
        A feature being "important" does NOT mean changing it will change the outcome. 
        Confounding variables, reverse causation, and selection bias may explain these patterns. 
        Do not make policy or intervention decisions based solely on these results.
        </div>
        """, unsafe_allow_html=True)

        # Feature Importance
        st.markdown("### Feature Importance Analysis")

        if hasattr(best_model, 'feature_importances_'):
            # Tree-based model
            importances = best_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df.head(15), x='Importance', y='Feature', orientation='h',
                        title=f"{best_name} - Feature Importance (Gini Importance)",
                        color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            <b>Interpretation:</b> Gini Importance measures how much each feature contributes to reducing impurity across all trees. 
            Higher values mean the feature is used more often in splits. However, this can be biased toward high-cardinality features.
            </div>
            """, unsafe_allow_html=True)

        elif hasattr(best_model, 'coef_'):
            # Linear model
            coefs = best_model.coef_
            if coefs.ndim > 1:
                coefs = coefs[0]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs,
                'Abs_Coefficient': np.abs(coefs)
            }).sort_values('Abs_Coefficient', ascending=False)

            fig = px.bar(importance_df.head(15), x='Coefficient', y='Feature', orientation='h',
                        title=f"{best_name} - Coefficients (Log-Odds for Classification)",
                        color='Coefficient', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            <b>Interpretation:</b> Coefficients represent the change in log-odds (classification) or target value (regression) 
            for a one-unit change in the feature, holding others constant. Positive = increase in outcome, Negative = decrease.
            </div>
            """, unsafe_allow_html=True)

        # SHAP Analysis (if available)
        if SHAP_AVAILABLE and best_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            st.markdown("### SHAP Explainability (SHapley Additive exPlanations)")
            with st.spinner("Computing SHAP values..."):
                try:
                    if task_type == 'classification':
                        explainer = shap.TreeExplainer(best_model)
                    else:
                        explainer = shap.TreeExplainer(best_model)

                    shap_values = explainer.shap_values(X_test.values[:100])  # Sample for speed

                    # Summary plot
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test.iloc[:100], feature_names=feature_names, show=False)
                    st.pyplot(fig)
                    plt.close()

                    st.markdown("""
                    <div class="info-box">
                    <b>SHAP Interpretation:</b> Each point represents a SHAP value for a feature in a specific prediction. 
                    Color (red/blue) = feature value (high/low). Position (left/right) = pushes prediction down/up. 
                    Features are ordered by total impact magnitude.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"SHAP computation failed: {e}. Using fallback interpretation.")

        # Practical Significance Deep Dive
        st.markdown("### Practical Significance Deep Dive")

        y_pred = best_model.predict(X_test)
        practical = evaluate_model_practical_significance(y_test, y_pred, task_type)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{practical["practical_rating"]}</div><div class="metric-label">Practical Rating</div></div>', unsafe_allow_html=True)
        with col2:
            if task_type == 'regression':
                st.markdown(f'<div class="metric-card"><div class="metric-value">{practical["MAPE"]:.1f}%</div><div class="metric-label">Mean Absolute % Error</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{practical["Lift_over_Baseline"]:.1f}%</div><div class="metric-label">Lift over Baseline</div></div>', unsafe_allow_html=True)
        with col3:
            if task_type == 'regression':
                st.markdown(f'<div class="metric-card"><div class="metric-value">{practical["R2"]:.3f}</div><div class="metric-label">R² (Variance Explained)</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{practical["Accuracy"]:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warning-box">
        <b>Business Context:</b><br>
        {practical['practical_message']}<br><br>
        {practical.get('variance_message', '')}
        </div>
        """, unsafe_allow_html=True)

        # Model Comparison Visualization
        st.markdown("### Model Comparison Dashboard")
        results_df = st.session_state['results_df']

        if task_type == 'regression':
            metrics_to_plot = ['RMSE', 'MAE', 'R2']
        else:
            metrics_to_plot = ['Accuracy', 'F1', 'Precision', 'Recall']

        fig = make_subplots(rows=1, cols=len(metrics_to_plot), subplot_titles=metrics_to_plot)
        for i, metric in enumerate(metrics_to_plot):
            if metric in results_df.columns:
                fig.add_trace(
                    go.Bar(x=results_df['Model'], y=results_df[metric], name=metric),
                    row=1, col=i+1
                )
        fig.update_layout(height=400, showlegend=False, title_text="Cross-Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Deployment Package
        st.markdown('<div class="sub-header">Stage 8: Model Registry & Deployment</div>', unsafe_allow_html=True)

        metrics = results_df.to_dict('records')
        assumptions = st.session_state.get('assumption_results', {})
        zip_data = create_zip_package(
            best_model, st.session_state['scaler'], st.session_state['encoders'],
            feature_names, target_col, metrics, assumptions
        )

        st.markdown(get_download_link(zip_data, "analytical_pipeline_v2.0.zip", "📦 Download Complete Pipeline Package (ZIP)"), unsafe_allow_html=True)
        st.markdown("""
        <div class="success-box">
        This package includes the trained model, preprocessing artifacts, feature definitions, assumption test results, and metadata. 
        Load in production using <code>metadata.json</code> to ensure consistency.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<center>Powered by Statistical Reasoning Engine v2.0 | Built for Analysts, Not Just Machines</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
