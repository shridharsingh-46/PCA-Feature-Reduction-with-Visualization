import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris PCA Explorer", page_icon="🌸", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d425c;
    }
    h1 {
        color: #ff4b4b;
        font-family: 'Outfit', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df, iris.feature_names, iris.target_names

df, feature_names, target_names = load_data()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

scale_data = st.sidebar.toggle("Use Standard Scaling", value=True, help="Centers and scales data to unit variance.")
n_components = st.sidebar.slider("Number of Principal Components", 1, 4, 3)

st.sidebar.markdown("---")
st.sidebar.info("Dimensions are reduced from 4 original features to the specified number of components.")

# --- PCA PROCESSING ---
X = df[feature_names]

if scale_data:
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X)
else:
    X_processed = X.values

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_processed)

# Create PCA DataFrame
pca_cols = [f"PC{i+1}" for i in range(n_components)]
pca_df = pd.DataFrame(X_pca, columns=pca_cols)
pca_df['species'] = df['species']

# --- HEADER ---
st.title("🌸 Iris PCA Explorer")
st.markdown("Interactive dimensionality reduction and forensic data analysis.")

# --- METRICS ---
cols = st.columns(min(n_components + 1, 4))
for i in range(min(n_components, 3)):
    with cols[i]:
        st.metric(f"PC{i+1} Var", f"{pca.explained_variance_ratio_[i]:.1%}")

with cols[-1]:
    total_var = sum(pca.explained_variance_ratio_)
    st.metric("Total Retained", f"{total_var:.1%}")

# --- TABS ---
tab_overview, tab_2d, tab_3d, tab_loadings = st.tabs([
    "📊 Dataset Overview", 
    "🎯 2D Visualization", 
    "🌐 3D Visualization", 
    "📈 Feature Loadings"
])

with tab_overview:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Raw Dataset")
        st.dataframe(df, use_container_width=True, height=400)
    
    with col2:
        st.subheader("Feature Correlations")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[feature_names].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

with tab_2d:
    st.subheader("2D Projection")
    if n_components >= 2:
        fig_2d = px.scatter(
            pca_df, x="PC1", y="PC2", color="species",
            title="PCA 2D Cluster Analysis",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={'PC1': f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", 
                    'PC2': f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"}
        )
        fig_2d.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='White')))
        st.plotly_chart(fig_2d, use_container_width=True)
    else:
        st.warning("Please select at least 2 components in the sidebar for 2D visualization.")

with tab_3d:
    st.subheader("3D Projection")
    if n_components >= 3:
        fig_3d = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3", color="species",
            title="PCA 3D Cluster Analysis",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={'PC1': f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", 
                    'PC2': f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                    'PC3': f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"}
        )
        fig_3d.update_layout(scene_zaxis_title_font_color="#ff4b4b")
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("Please select at least 3 components in the sidebar for 3D visualization.")

with tab_loadings:
    st.subheader("Feature Importance (Loadings)")
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=pca_cols, 
        index=feature_names
    )
    
    selected_pc = st.selectbox("Select Component to Analyze", pca_cols)
    
    fig_loadings = px.bar(
        loadings, x=loadings.index, y=selected_pc,
        title=f"Original Feature Contribution to {selected_pc}",
        template="plotly_dark",
        labels={'index': 'Feature', selected_pc: 'Loading Score'},
        color=selected_pc,
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_loadings, use_container_width=True)
    
    st.info("Positive or negative scores indicate how much a feature pulls the component in that direction.")
