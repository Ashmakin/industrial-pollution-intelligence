"""
Principal Component Analysis for Water Quality Data

Implements PCA, t-SNE, and UMAP for dimensionality reduction and pattern discovery
in water quality datasets.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityPCA:
    """Principal Component Analysis for water quality data"""
    
    def __init__(self, n_components: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit PCA and transform data"""
        
                                   
        X = df[feature_columns].dropna()
        
        if len(X) == 0:
            raise ValueError("No valid data points found")
        
                        
        X_scaled = self.scaler.fit_transform(X)
        
                 
        X_pca = self.pca.fit_transform(X_scaled)
        
                       
        self.feature_names = feature_columns
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_
        
                           
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=X.index
        )
        
                      
        metadata_columns = ['monitoring_time', 'station_name', 'province', 'watershed', 'water_quality_grade']
        available_metadata = [col for col in metadata_columns if col in df.columns]
        
        for col in available_metadata:
            pca_df[col] = df.loc[X.index, col].values
        
                              
        components_df = pd.DataFrame(
            self.components_,
            columns=feature_columns,
            index=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        logger.info(f"PCA completed. Explained variance: {self.explained_variance_ratio_.sum():.3f}")
        
        return pca_df, components_df
    
    def get_feature_importance(self, n_components: int = 5) -> pd.DataFrame:
        """Get feature importance for top principal components"""
        
        if self.components_ is None:
            raise ValueError("PCA not fitted yet")
        
        importance_df = pd.DataFrame()
        
        for i in range(min(n_components, len(self.components_))):
            pc_name = f'PC{i+1}'
            explained_var = self.explained_variance_ratio_[i]
            
                                                     
            importance = np.abs(self.components_[i])
            
                                             
            weighted_importance = importance * explained_var
            
            importance_df[pc_name] = pd.Series(
                weighted_importance,
                index=self.feature_names
            )
        
                                  
        importance_df['total_importance'] = importance_df.sum(axis=1)
        importance_df = importance_df.sort_values('total_importance', ascending=False)
        
        return importance_df
    
    def plot_explained_variance(self, figsize: Tuple[int, int] = (12, 6)) -> go.Figure:
        """Plot explained variance ratio"""
        
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA not fitted yet")
        
                                       
        cumsum = np.cumsum(self.explained_variance_ratio_)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Individual Explained Variance', 'Cumulative Explained Variance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
                                       
        fig.add_trace(
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(self.explained_variance_ratio_))],
                y=self.explained_variance_ratio_,
                name='Individual',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
                                       
        fig.add_trace(
            go.Scatter(
                x=[f'PC{i+1}' for i in range(len(cumsum))],
                y=cumsum,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='PCA Explained Variance Analysis',
            height=figsize[1],
            width=figsize[0],
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Principal Component", row=1, col=1)
        fig.update_xaxes(title_text="Principal Component", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Explained Variance", row=1, col=2)
        
        return fig
    
    def plot_biplot(self, pca_df: pd.DataFrame, components_df: pd.DataFrame, 
                   pc1: int = 1, pc2: int = 2, max_features: int = 10) -> go.Figure:
        """Create PCA biplot showing scores and loadings"""
        
        pc1_name = f'PC{pc1}'
        pc2_name = f'PC{pc2}'
        
                                           
        pc1_loadings = np.abs(components_df.loc[pc1_name])
        pc2_loadings = np.abs(components_df.loc[pc2_name])
        
                             
        combined_importance = np.sqrt(pc1_loadings**2 + pc2_loadings**2)
        top_features = combined_importance.nlargest(max_features).index.tolist()
        
                                        
        fig = go.Figure()
        
                                                   
        if 'water_quality_grade' in pca_df.columns:
            grades = pca_df['water_quality_grade'].fillna('Unknown')
            colors = px.colors.qualitative.Set1[:len(grades.unique())]
            color_map = dict(zip(grades.unique(), colors))
            
            for grade in grades.unique():
                mask = grades == grade
                fig.add_trace(
                    go.Scatter(
                        x=pca_df.loc[mask, pc1_name],
                        y=pca_df.loc[mask, pc2_name],
                        mode='markers',
                        name=f'Grade {grade}',
                        marker=dict(
                            color=color_map[grade],
                            size=6,
                            opacity=0.7
                        ),
                        text=pca_df.loc[mask, 'station_name'],
                        hovertemplate='<b>%{text}</b><br>' +
                                    f'{pc1_name}: %{{x:.3f}}<br>' +
                                    f'{pc2_name}: %{{y:.3f}}<br>' +
                                    '<extra></extra>'
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=pca_df[pc1_name],
                    y=pca_df[pc2_name],
                    mode='markers',
                    name='Data Points',
                    marker=dict(size=6, opacity=0.7),
                    text=pca_df['station_name'],
                    hovertemplate='<b>%{text}</b><br>' +
                                f'{pc1_name}: %{{x:.3f}}<br>' +
                                f'{pc2_name}: %{{y:.3f}}<br>' +
                                '<extra></extra>'
                )
            )
        
                             
        for feature in top_features:
            x0, y0 = 0, 0
            x1 = components_df.loc[pc1_name, feature] * 3                        
            y1 = components_df.loc[pc2_name, feature] * 3
            
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines+markers',
                    name=f'{feature}',
                    line=dict(color='red', width=2),
                    marker=dict(size=8, color='red'),
                    showlegend=False,
                    hovertemplate=f'<b>{feature}</b><br>' +
                                f'{pc1_name}: {x1:.3f}<br>' +
                                f'{pc2_name}: {y1:.3f}<br>' +
                                '<extra></extra>'
                )
            )
            
                                
            fig.add_annotation(
                x=x1,
                y=y1,
                text=feature,
                showarrow=False,
                font=dict(size=10, color='red'),
                xshift=10,
                yshift=10
            )
        
        fig.update_layout(
            title=f'PCA Biplot: {pc1_name} vs {pc2_name}',
            xaxis_title=f'{pc1_name} ({self.explained_variance_ratio_[pc1-1]:.1%} variance)',
            yaxis_title=f'{pc2_name} ({self.explained_variance_ratio_[pc2-1]:.1%} variance)',
            width=800,
            height=600
        )
        
        return fig
    
    def cluster_analysis(self, pca_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Perform clustering on PCA-transformed data"""
        
                                          
        pca_features = [col for col in pca_df.columns if col.startswith('PC')]
        X_pca = pca_df[pca_features].values
        
                            
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(X_pca)
        
                            
        pca_df_clustered = pca_df.copy()
        pca_df_clustered['cluster'] = clusters
        
                                         
        cluster_summary = pca_df_clustered.groupby('cluster')[pca_features].mean()
        
        logger.info(f"Clustering completed. {n_clusters} clusters identified")
        
        return pca_df_clustered, cluster_summary

class WaterQualityDimensionalityReduction:
    """Comprehensive dimensionality reduction analysis"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_analyzer = WaterQualityPCA(random_state=random_state)
        self.tsne = None
        self.umap_reducer = None
        
    def run_comprehensive_analysis(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """Run PCA, t-SNE, and UMAP analysis"""
        
        results = {}
        
                      
        logger.info("Running PCA analysis...")
        pca_df, components_df = self.pca_analyzer.fit_transform(df, feature_columns)
        results['pca'] = {
            'data': pca_df,
            'components': components_df,
            'explained_variance_ratio': self.pca_analyzer.explained_variance_ratio_
        }
        
                        
        logger.info("Running t-SNE analysis...")
        X_scaled = self.pca_analyzer.scaler.transform(df[feature_columns].dropna())
        self.tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
        tsne_results = self.tsne.fit_transform(X_scaled)
        
        tsne_df = pd.DataFrame(
            tsne_results,
            columns=['tSNE1', 'tSNE2'],
            index=df[feature_columns].dropna().index
        )
        
                      
        metadata_columns = ['monitoring_time', 'station_name', 'province', 'watershed']
        available_metadata = [col for col in metadata_columns if col in df.columns]
        
        for col in available_metadata:
            tsne_df[col] = df.loc[tsne_df.index, col].values
        
        results['tsne'] = {'data': tsne_df}
        
                       
        logger.info("Running UMAP analysis...")
        self.umap_reducer = umap.UMAP(n_components=2, random_state=self.random_state)
        umap_results = self.umap_reducer.fit_transform(X_scaled)
        
        umap_df = pd.DataFrame(
            umap_results,
            columns=['UMAP1', 'UMAP2'],
            index=df[feature_columns].dropna().index
        )
        
                      
        for col in available_metadata:
            umap_df[col] = df.loc[umap_df.index, col].values
        
        results['umap'] = {'data': umap_df}
        
        return results
    
    def plot_comparison(self, results: Dict) -> go.Figure:
        """Plot comparison of PCA, t-SNE, and UMAP results"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA (PC1 vs PC2)', 'PCA (PC3 vs PC4)', 't-SNE', 'UMAP'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
                   
        pca_df = results['pca']['data']
        
                    
        fig.add_trace(
            go.Scatter(
                x=pca_df['PC1'],
                y=pca_df['PC2'],
                mode='markers',
                name='PCA 1-2',
                marker=dict(size=4, opacity=0.7),
                text=pca_df['station_name'],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
                    
        if 'PC3' in pca_df.columns and 'PC4' in pca_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pca_df['PC3'],
                    y=pca_df['PC4'],
                    mode='markers',
                    name='PCA 3-4',
                    marker=dict(size=4, opacity=0.7),
                    text=pca_df['station_name'],
                    hovertemplate='<b>%{text}</b><br>PC3: %{x:.3f}<br>PC4: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
        
                    
        tsne_df = results['tsne']['data']
        fig.add_trace(
            go.Scatter(
                x=tsne_df['tSNE1'],
                y=tsne_df['tSNE2'],
                mode='markers',
                name='t-SNE',
                marker=dict(size=4, opacity=0.7),
                text=tsne_df['station_name'],
                hovertemplate='<b>%{text}</b><br>tSNE1: %{x:.3f}<br>tSNE2: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
                   
        umap_df = results['umap']['data']
        fig.add_trace(
            go.Scatter(
                x=umap_df['UMAP1'],
                y=umap_df['UMAP2'],
                mode='markers',
                name='UMAP',
                marker=dict(size=4, opacity=0.7),
                text=umap_df['station_name'],
                hovertemplate='<b>%{text}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Dimensionality Reduction Comparison',
            height=800,
            width=1000,
            showlegend=False
        )
        
        return fig

def main():
    """Example usage"""
                         
    df = pd.read_parquet("data/processed_water_quality.parquet")
    
                            
    feature_columns = [
        'temperature', 'ph', 'dissolved_oxygen', 'conductivity', 'turbidity',
        'permanganate_index', 'ammonia_nitrogen', 'total_phosphorus', 'total_nitrogen',
        'chlorophyll_a', 'algae_density'
    ]
    
                               
    available_features = [col for col in feature_columns if col in df.columns]
    
                                
    analyzer = WaterQualityDimensionalityReduction()
    results = analyzer.run_comprehensive_analysis(df, available_features)
    
                           
    pca_analyzer = WaterQualityPCA()
    pca_df, components_df = pca_analyzer.fit_transform(df, available_features)
    
                             
    var_plot = pca_analyzer.plot_explained_variance()
    var_plot.show()
    
                 
    biplot = pca_analyzer.plot_biplot(pca_df, components_df)
    biplot.show()
    
                     
    comparison_plot = analyzer.plot_comparison(results)
    comparison_plot.show()
    
    print("Dimensionality reduction analysis completed")

if __name__ == "__main__":
    main()