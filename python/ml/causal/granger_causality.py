"""
Granger Causality Analysis for Water Quality Data

Tests causal relationships between different water quality parameters
and environmental factors.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from typing import List, Dict, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityGrangerCausality:
    """Granger causality analysis for water quality time series"""
    
    def __init__(self, max_lags: int = 12, significance_level: float = 0.05):
        self.max_lags = max_lags
        self.significance_level = significance_level
        self.results = {}
        
    def prepare_stationary_data(self, df: pd.DataFrame, variables: List[str], 
                               station: str) -> pd.DataFrame:
        """Prepare stationary time series data for Granger causality tests"""
        
        station_data = df[df['station_name'] == station].copy()
        station_data = station_data.sort_values('monitoring_time')
        
                                                    
        data = station_data[variables].dropna()
        
        if len(data) < 50:                        
            return pd.DataFrame()
        
                                               
        stationary_data = data.copy()
        
        for var in variables:
                                                    
            from statsmodels.tsa.stattools import adfuller
            
                                  
            adf_result = adfuller(data[var].dropna())
            
            if adf_result[1] > 0.05:                  
                                      
                diff_data = data[var].diff().dropna()
                adf_diff = adfuller(diff_data)
                
                if adf_diff[1] <= 0.05:                                  
                    stationary_data[var] = diff_data
                    logger.info(f"Applied first difference to {var} for station {station}")
                else:
                    logger.warning(f"Could not make {var} stationary for station {station}")
        
        return stationary_data
    
    def test_granger_causality(self, df: pd.DataFrame, cause_var: str, effect_var: str, 
                              station: str) -> Dict:
        """Test Granger causality between two variables for a specific station"""
        
                      
        variables = [cause_var, effect_var]
        data = self.prepare_stationary_data(df, variables, station)
        
        if len(data) < 30:
            return {'error': 'Insufficient data'}
        
        try:
                                            
            test_result = grangercausalitytests(
                data[[effect_var, cause_var]],                                            
                maxlag=self.max_lags,
                verbose=False
            )
            
                                                  
            results = {
                'station': station,
                'cause_var': cause_var,
                'effect_var': effect_var,
                'lags': {},
                'summary': {}
            }
            
                                          
            for lag in range(1, min(self.max_lags + 1, len(test_result))):
                lag_result = test_result[lag]
                
                                                      
                ssr_ftest = lag_result[0]['ssr_ftest']
                ssr_chi2test = lag_result[0]['ssr_chi2test']
                lrtest = lag_result[0]['lrtest']
                params_ftest = lag_result[0]['params_ftest']
                
                results['lags'][lag] = {
                    'ssr_ftest': {
                        'statistic': ssr_ftest[0],
                        'p_value': ssr_ftest[1],
                        'critical_values': ssr_ftest[2]
                    },
                    'ssr_chi2test': {
                        'statistic': ssr_chi2test[0],
                        'p_value': ssr_chi2test[1]
                    },
                    'lrtest': {
                        'statistic': lrtest[0],
                        'p_value': lrtest[1]
                    },
                    'params_ftest': {
                        'statistic': params_ftest[0],
                        'p_value': params_ftest[1]
                    }
                }
            
                                            
            var_model = VAR(data)
            lag_order = var_model.select_order(maxlags=self.max_lags)
            
            results['summary'] = {
                'aic_lag': lag_order.aic,
                'bic_lag': lag_order.bic,
                'hqic_lag': lag_order.hqic,
                'fpe_lag': lag_order.fpe
            }
            
                                           
            best_lag = lag_order.aic
            if best_lag in results['lags']:
                best_result = results['lags'][best_lag]
                results['summary']['granger_causality'] = (
                    best_result['ssr_ftest']['p_value'] < self.significance_level
                )
                results['summary']['p_value'] = best_result['ssr_ftest']['p_value']
                results['summary']['f_statistic'] = best_result['ssr_ftest']['statistic']
            else:
                results['summary']['granger_causality'] = False
                results['summary']['p_value'] = 1.0
                results['summary']['f_statistic'] = 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Granger causality test for {station}: {e}")
            return {'error': str(e)}
    
    def analyze_all_stations(self, df: pd.DataFrame, variable_pairs: List[Tuple[str, str]]) -> Dict:
        """Analyze Granger causality for all stations and variable pairs"""
        
        results = {}
        stations = df['station_name'].unique()
        
        for cause_var, effect_var in variable_pairs:
            logger.info(f"Testing causality: {cause_var} -> {effect_var}")
            
            results[f"{cause_var}_causes_{effect_var}"] = {}
            
            for station in stations:
                station_result = self.test_granger_causality(df, cause_var, effect_var, station)
                
                if 'error' not in station_result:
                    results[f"{cause_var}_causes_{effect_var}"][station] = station_result
                else:
                    logger.warning(f"Skipping {station}: {station_result['error']}")
        
        self.results = results
        return results
    
    def summarize_results(self) -> pd.DataFrame:
        """Summarize Granger causality results across all stations"""
        
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for causality_key, station_results in self.results.items():
            cause_var, effect_var = causality_key.split('_causes_')
            
            for station, result in station_results.items():
                if 'summary' in result:
                    summary_data.append({
                        'station': station,
                        'cause_variable': cause_var,
                        'effect_variable': effect_var,
                        'granger_causality': result['summary'].get('granger_causality', False),
                        'p_value': result['summary'].get('p_value', 1.0),
                        'f_statistic': result['summary'].get('f_statistic', 0.0),
                        'optimal_lag': result['summary'].get('aic_lag', 0)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        if len(summary_df) > 0:
                                     
            summary_df['significance'] = summary_df['p_value'].apply(
                lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
            )
        
        return summary_df
    
    def plot_causality_network(self, min_significance: float = 0.05) -> Dict:
        """Create network visualization of significant causal relationships"""
        
        summary_df = self.summarize_results()
        
        if len(summary_df) == 0:
            return {}
        
                                          
        significant_df = summary_df[
            (summary_df['p_value'] < min_significance) & 
            (summary_df['granger_causality'] == True)
        ]
        
        if len(significant_df) == 0:
            logger.info("No significant Granger causality relationships found")
            return {}
        
                             
        nodes = set()
        edges = []
        
        for _, row in significant_df.iterrows():
            cause = row['cause_variable']
            effect = row['effect_variable']
            
            nodes.add(cause)
            nodes.add(effect)
            
            edges.append({
                'source': cause,
                'target': effect,
                'strength': -np.log10(row['p_value']),                               
                'station_count': 1                      
            })
        
                                         
        edge_dict = {}
        for edge in edges:
            key = (edge['source'], edge['target'])
            if key in edge_dict:
                edge_dict[key]['strength'] += edge['strength']
                edge_dict[key]['station_count'] += 1
            else:
                edge_dict[key] = edge
        
        network_data = {
            'nodes': list(nodes),
            'edges': list(edge_dict.values()),
            'summary': {
                'total_relationships': len(significant_df),
                'unique_variable_pairs': len(edge_dict),
                'variables_involved': len(nodes)
            }
        }
        
        return network_data
    
    def analyze_causal_strength(self) -> pd.DataFrame:
        """Analyze the strength and consistency of causal relationships"""
        
        summary_df = self.summarize_results()
        
        if len(summary_df) == 0:
            return pd.DataFrame()
        
                                 
        causal_strength = summary_df.groupby(['cause_variable', 'effect_variable']).agg({
            'granger_causality': ['sum', 'count'],
            'p_value': ['mean', 'min'],
            'f_statistic': ['mean', 'max']
        }).round(4)
        
                              
        causal_strength.columns = ['_'.join(col).strip() for col in causal_strength.columns]
        causal_strength = causal_strength.reset_index()
        
                                       
        causal_strength['consistency_ratio'] = (
            causal_strength['granger_causality_sum'] / causal_strength['granger_causality_count']
        )
        causal_strength['strength_score'] = (
            causal_strength['consistency_ratio'] * (1 - causal_strength['p_value_mean'])
        )
        
                                
        causal_strength = causal_strength.sort_values('strength_score', ascending=False)
        
        return causal_strength

class WaterQualityCausalInference:
    """Comprehensive causal inference analysis for water quality"""
    
    def __init__(self):
        self.granger_analyzer = WaterQualityGrangerCausality()
        
    def run_comprehensive_analysis(self, df: pd.DataFrame, 
                                 variable_pairs: List[Tuple[str, str]]) -> Dict:
        """Run comprehensive causal inference analysis"""
        
        results = {}
        
                                    
        logger.info("Running Granger causality analysis...")
        granger_results = self.granger_analyzer.analyze_all_stations(df, variable_pairs)
        results['granger_causality'] = granger_results
        
                            
        results['summary'] = self.granger_analyzer.summarize_results()
        
                                  
        results['causal_strength'] = self.granger_analyzer.analyze_causal_strength()
        
                          
        results['causality_network'] = self.granger_analyzer.plot_causality_network()
        
        return results
    
    def identify_key_causal_paths(self, min_consistency: float = 0.3) -> List[Dict]:
        """Identify key causal pathways in water quality data"""
        
        causal_strength = self.granger_analyzer.analyze_causal_strength()
        
        if len(causal_strength) == 0:
            return []
        
                                             
        consistent_relationships = causal_strength[
            causal_strength['consistency_ratio'] >= min_consistency
        ]
        
        key_paths = []
        
        for _, row in consistent_relationships.iterrows():
            key_paths.append({
                'cause': row['cause_variable'],
                'effect': row['effect_variable'],
                'consistency': row['consistency_ratio'],
                'strength': row['strength_score'],
                'avg_p_value': row['p_value_mean'],
                'station_count': row['granger_causality_count']
            })
        
        return sorted(key_paths, key=lambda x: x['strength'], reverse=True)

def main():
    """Example usage"""
                         
    df = pd.read_parquet("data/processed_water_quality.parquet")
    
                                                 
    variable_pairs = [
        ('ammonia_nitrogen', 'total_phosphorus'),
        ('total_phosphorus', 'chlorophyll_a'),
        ('total_nitrogen', 'chlorophyll_a'),
        ('temperature', 'dissolved_oxygen'),
        ('ph', 'ammonia_nitrogen'),
        ('turbidity', 'algae_density'),
        ('conductivity', 'total_phosphorus')
    ]
    
                                   
    causal_analyzer = WaterQualityCausalInference()
    results = causal_analyzer.run_comprehensive_analysis(df, variable_pairs)
    
                        
    key_paths = causal_analyzer.identify_key_causal_paths(min_consistency=0.2)
    
    print("Key Causal Pathways:")
    for path in key_paths[:5]:         
        print(f"{path['cause']} -> {path['effect']}: "
              f"Consistency={path['consistency']:.2f}, "
              f"Strength={path['strength']:.3f}")
    
    print("Causal inference analysis completed")

if __name__ == "__main__":
    main()

