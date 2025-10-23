"""
Product Lifecycle Pollution Tracking Module

Tracks pollution generation across smartphone manufacturing supply chain
and correlates with water quality monitoring data.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PollutionImpact:
    """Data structure for pollution impact at each manufacturing stage"""
    stage: str
    location: str
    pollutants: Dict[str, float]                            
    water_usage: float                   
    energy_consumption: float                
    waste_generation: float               

class SmartphoneLifecycleTracker:
    """Tracks pollution across smartphone manufacturing lifecycle"""
    
    def __init__(self):
        self.pollution_impacts = self._initialize_pollution_data()
        self.industrial_zones = self._initialize_industrial_zones()
        self.supply_chain_graph = self._build_supply_chain_graph()
        
    def _initialize_pollution_data(self) -> List[PollutionImpact]:
        """Initialize pollution impact data for each manufacturing stage"""
        
        pollution_data = [
                                     
            PollutionImpact(
                stage="Mining - Rare Earth Elements",
                location="Inner Mongolia, China",
                pollutants={
                    "heavy_metals": 0.05,                
                    "radioactive_waste": 0.01,
                    "acid_mine_drainage": 0.1,
                    "sediment": 0.5
                },
                water_usage=1000,                    
                energy_consumption=50,
                waste_generation=10
            ),
            
                                     
            PollutionImpact(
                stage="Circuit Board Manufacturing",
                location="Shenzhen, Guangdong",
                pollutants={
                    "copper": 0.02,
                    "lead": 0.005,
                    "cadmium": 0.001,
                    "organic_solvents": 0.01,
                    "etching_waste": 0.05
                },
                water_usage=200,
                energy_consumption=25,
                waste_generation=0.5
            ),
            
            PollutionImpact(
                stage="Display Manufacturing",
                location="Suzhou, Jiangsu",
                pollutants={
                    "fluorides": 0.003,
                    "organic_solvents": 0.008,
                    "heavy_metals": 0.002,
                    "vocs": 0.005
                },
                water_usage=150,
                energy_consumption=30,
                waste_generation=0.3
            ),
            
            PollutionImpact(
                stage="Battery Manufacturing",
                location="Ningde, Fujian",
                pollutants={
                    "lithium": 0.01,
                    "cobalt": 0.008,
                    "nickel": 0.005,
                    "organic_solvents": 0.003,
                    "electrolyte": 0.002
                },
                water_usage=100,
                energy_consumption=40,
                waste_generation=0.2
            ),
            
            PollutionImpact(
                stage="Camera Module Assembly",
                location="Dongguan, Guangdong",
                pollutants={
                    "rare_earth_metals": 0.001,
                    "organic_solvents": 0.002,
                    "adhesives": 0.001
                },
                water_usage=50,
                energy_consumption=15,
                waste_generation=0.1
            ),
            
                            
            PollutionImpact(
                stage="Device Assembly",
                location="Zhengzhou, Henan",
                pollutants={
                    "vocs": 0.01,
                    "adhesives": 0.005,
                    "cleaning_solvents": 0.003
                },
                water_usage=80,
                energy_consumption=20,
                waste_generation=0.15
            ),
            
                       
            PollutionImpact(
                stage="Packaging",
                location="Various",
                pollutants={
                    "plastic_waste": 0.1,
                    "paper_waste": 0.05,
                    "ink_waste": 0.001
                },
                water_usage=30,
                energy_consumption=10,
                waste_generation=0.2
            )
        ]
        
        return pollution_data
    
    def _initialize_industrial_zones(self) -> Dict[str, Dict]:
        """Initialize industrial zone data with coordinates and characteristics"""
        
        zones = {
            "Shenzhen": {
                "coordinates": (22.5431, 114.0579),
                "province": "Guangdong",
                "area_id": "440000",
                "industries": ["electronics", "semiconductor", "circuit_boards"],
                "water_bodies": ["Shenzhen Bay", "Pearl River Delta"],
                "monitoring_stations": ["深圳湾", "珠江口", "大鹏湾"]
            },
            "Shanghai": {
                "coordinates": (31.2304, 121.4737),
                "province": "Shanghai",
                "area_id": "310000",
                "industries": ["electronics", "automotive", "chemicals"],
                "water_bodies": ["Huangpu River", "Yangtze River", "East China Sea"],
                "monitoring_stations": ["黄浦江", "长江口", "东海"]
            },
            "Suzhou": {
                "coordinates": (31.2989, 120.5853),
                "province": "Jiangsu",
                "area_id": "320000",
                "industries": ["electronics", "display", "semiconductor"],
                "water_bodies": ["Taihu Lake", "Yangtze River"],
                "monitoring_stations": ["太湖", "长江", "京杭大运河"]
            },
            "Ningde": {
                "coordinates": (26.6592, 119.5233),
                "province": "Fujian",
                "area_id": "350000",
                "industries": ["battery", "new_energy", "chemicals"],
                "water_bodies": ["Minjiang River", "East China Sea"],
                "monitoring_stations": ["闽江", "东海", "宁德湾"]
            },
            "Zhengzhou": {
                "coordinates": (34.7466, 113.6254),
                "province": "Henan",
                "area_id": "410000",
                "industries": ["assembly", "manufacturing", "logistics"],
                "water_bodies": ["Yellow River", "Jialu River"],
                "monitoring_stations": ["黄河", "贾鲁河", "郑东新区"]
            }
        }
        
        return zones
    
    def _build_supply_chain_graph(self) -> nx.DiGraph:
        """Build directed graph representing smartphone supply chain"""
        
        G = nx.DiGraph()
        
                                          
        for impact in self.pollution_impacts:
            G.add_node(
                impact.stage,
                location=impact.location,
                pollutants=impact.pollutants,
                water_usage=impact.water_usage,
                energy_consumption=impact.energy_consumption,
                waste_generation=impact.waste_generation
            )
        
                                   
        supply_chain_flow = [
            ("Mining - Rare Earth Elements", "Circuit Board Manufacturing"),
            ("Mining - Rare Earth Elements", "Display Manufacturing"),
            ("Mining - Rare Earth Elements", "Camera Module Assembly"),
            ("Circuit Board Manufacturing", "Device Assembly"),
            ("Display Manufacturing", "Device Assembly"),
            ("Battery Manufacturing", "Device Assembly"),
            ("Camera Module Assembly", "Device Assembly"),
            ("Device Assembly", "Packaging")
        ]
        
        for source, target in supply_chain_flow:
            G.add_edge(source, target, material_flow=1.0)
        
        return G
    
    def calculate_total_pollution_per_phone(self) -> Dict[str, float]:
        """Calculate total pollution impact per smartphone unit"""
        
        total_pollution = {}
        total_water = 0
        total_energy = 0
        total_waste = 0
        
        for impact in self.pollution_impacts:
                            
            for pollutant, amount in impact.pollutants.items():
                total_pollution[pollutant] = total_pollution.get(pollutant, 0) + amount
            
                               
            total_water += impact.water_usage
            total_energy += impact.energy_consumption
            total_waste += impact.waste_generation
        
        total_pollution["water_usage"] = total_water
        total_pollution["energy_consumption"] = total_energy
        total_pollution["waste_generation"] = total_waste
        
        return total_pollution
    
    def estimate_regional_pollution_impact(self, production_volume: int = 1000000) -> Dict[str, Dict]:
        """Estimate pollution impact by region based on production volume"""
        
        regional_impact = {}
        
        for impact in self.pollution_impacts:
            location = impact.location
            if location not in regional_impact:
                regional_impact[location] = {
                    "pollutants": {},
                    "water_usage": 0,
                    "energy_consumption": 0,
                    "waste_generation": 0
                }
            
                                        
            for pollutant, amount in impact.pollutants.items():
                regional_impact[location]["pollutants"][pollutant] = (
                    regional_impact[location]["pollutants"].get(pollutant, 0) + 
                    amount * production_volume
                )
            
            regional_impact[location]["water_usage"] += impact.water_usage * production_volume
            regional_impact[location]["energy_consumption"] += impact.energy_consumption * production_volume
            regional_impact[location]["waste_generation"] += impact.waste_generation * production_volume
        
        return regional_impact
    
    def correlate_with_water_quality_data(self, water_quality_df: pd.DataFrame, 
                                        time_window_days: int = 30) -> pd.DataFrame:
        """Correlate manufacturing pollution with water quality monitoring data"""
        
        correlations = []
        
        for zone_name, zone_data in self.industrial_zones.items():
            area_id = zone_data["area_id"]
            monitoring_stations = zone_data["monitoring_stations"]
            
                                                       
            zone_water_data = water_quality_df[
                (water_quality_df['area_id'] == area_id) |
                (water_quality_df['station_name'].isin(monitoring_stations))
            ]
            
            if len(zone_water_data) == 0:
                continue
            
                                                   
            zone_manufacturing = [
                impact for impact in self.pollution_impacts 
                if zone_name in impact.location
            ]
            
            for manufacturing_stage in zone_manufacturing:
                                                                                  
                zone_water_data_copy = zone_water_data.copy()
                zone_water_data_copy['manufacturing_intensity'] = self._simulate_production_schedule(
                    zone_water_data_copy['monitoring_time']
                )
                
                                                                                         
                pollution_indicators = ['ammonia_nitrogen', 'total_phosphorus', 'conductivity', 'turbidity']
                
                for indicator in pollution_indicators:
                    if indicator in zone_water_data_copy.columns:
                        correlation = zone_water_data_copy['manufacturing_intensity'].corr(
                            zone_water_data_copy[indicator]
                        )
                        
                        correlations.append({
                            'zone': zone_name,
                            'manufacturing_stage': manufacturing_stage.stage,
                            'pollution_indicator': indicator,
                            'correlation': correlation,
                            'station_count': len(zone_water_data['station_name'].unique()),
                            'data_points': len(zone_water_data)
                        })
        
        return pd.DataFrame(correlations)
    
    def _simulate_production_schedule(self, timestamps: pd.Series) -> pd.Series:
        """Simulate manufacturing production schedule based on timestamps"""
        
        production_intensity = []
        
        for timestamp in timestamps:
                                                                    
            weekday = timestamp.weekday()
            hour = timestamp.hour
            
                                                
            base_intensity = 1.0 if weekday < 5 else 0.3
            
                                                                 
            if 8 <= hour <= 17:
                hourly_factor = 1.0
            elif 18 <= hour <= 22:
                hourly_factor = 0.8                 
            elif 23 <= hour or hour <= 7:
                hourly_factor = 0.6               
            else:
                hourly_factor = 0.5
            
                                       
            random_factor = np.random.normal(1.0, 0.1)
            
            intensity = base_intensity * hourly_factor * random_factor
            production_intensity.append(max(0, intensity))
        
        return pd.Series(production_intensity, index=timestamps.index)
    
    def generate_pollution_signature(self, manufacturing_stage: str) -> Dict:
        """Generate pollution signature for a specific manufacturing stage"""
        
        stage_data = next(
            (impact for impact in self.pollution_impacts if impact.stage == manufacturing_stage),
            None
        )
        
        if not stage_data:
            return {}
        
                                                                 
        signature = {
            'stage': manufacturing_stage,
            'location': stage_data.location,
            'pollution_profile': {},
            'environmental_impact': {
                'water_usage': stage_data.water_usage,
                'energy_consumption': stage_data.energy_consumption,
                'waste_generation': stage_data.waste_generation
            }
        }
        
                                            
        total_pollution = sum(stage_data.pollutants.values())
        for pollutant, amount in stage_data.pollutants.items():
            signature['pollution_profile'][pollutant] = {
                'concentration': amount,
                'relative_concentration': amount / total_pollution if total_pollution > 0 else 0,
                'risk_level': self._assess_risk_level(pollutant, amount)
            }
        
        return signature
    
    def _assess_risk_level(self, pollutant: str, concentration: float) -> str:
        """Assess risk level of pollutant based on concentration"""
        
                                        
        risk_thresholds = {
            'heavy_metals': 0.05,
            'radioactive_waste': 0.01,
            'organic_solvents': 0.02,
            'lithium': 0.01,
            'cobalt': 0.01,
            'copper': 0.05,
            'lead': 0.01,
            'cadmium': 0.005,
            'fluorides': 0.01
        }
        
        threshold = risk_thresholds.get(pollutant, 0.01)
        
        if concentration >= threshold:
            return 'high'
        elif concentration >= threshold * 0.5:
            return 'medium'
        else:
            return 'low'
    
    def create_supply_chain_network_data(self) -> Dict:
        """Create network data for D3.js visualization"""
        
        nodes = []
        links = []
        
                   
        for impact in self.pollution_impacts:
            total_pollution = sum(impact.pollutants.values())
            
            nodes.append({
                'id': impact.stage,
                'name': impact.stage,
                'location': impact.location,
                'pollution_intensity': total_pollution,
                'water_usage': impact.water_usage,
                'energy_consumption': impact.energy_consumption,
                'waste_generation': impact.waste_generation,
                'pollutants': impact.pollutants
            })
        
                   
        for edge in self.supply_chain_graph.edges(data=True):
            links.append({
                'source': edge[0],
                'target': edge[1],
                'material_flow': edge[2]['material_flow']
            })
        
        return {
            'nodes': nodes,
            'links': links,
            'metadata': {
                'total_nodes': len(nodes),
                'total_links': len(links),
                'total_pollution_per_phone': self.calculate_total_pollution_per_phone()
            }
        }

class WaterQualityCorrelationAnalyzer:
    """Analyzes correlation between product lifecycle and water quality"""
    
    def __init__(self, lifecycle_tracker: SmartphoneLifecycleTracker):
        self.lifecycle_tracker = lifecycle_tracker
    
    def analyze_temporal_correlations(self, water_quality_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze temporal correlations between manufacturing and water quality"""
        
        correlations_df = self.lifecycle_tracker.correlate_with_water_quality_data(water_quality_df)
        
                                      
        correlations_df['significant'] = correlations_df['correlation'].abs() > 0.3
        correlations_df['strength'] = correlations_df['correlation'].abs().apply(
            lambda x: 'strong' if x > 0.5 else 'moderate' if x > 0.3 else 'weak'
        )
        
        return correlations_df
    
    def identify_pollution_hotspots(self, water_quality_df: pd.DataFrame) -> List[Dict]:
        """Identify areas with highest pollution correlation"""
        
        correlations_df = self.analyze_temporal_correlations(water_quality_df)
        
                                                         
        zone_correlations = correlations_df.groupby('zone').agg({
            'correlation': ['mean', 'max', 'count'],
            'significant': 'sum'
        }).round(3)
        
        zone_correlations.columns = ['avg_correlation', 'max_correlation', 'measurement_count', 'significant_count']
        zone_correlations = zone_correlations.reset_index()
        
                                                    
        hotspots = zone_correlations[
            (zone_correlations['avg_correlation'] > 0.2) |
            (zone_correlations['max_correlation'] > 0.4)
        ].sort_values('avg_correlation', ascending=False)
        
        hotspot_list = []
        for _, row in hotspots.iterrows():
            zone_data = self.lifecycle_tracker.industrial_zones.get(row['zone'], {})
            
            hotspot_list.append({
                'zone': row['zone'],
                'coordinates': zone_data.get('coordinates', (0, 0)),
                'avg_correlation': row['avg_correlation'],
                'max_correlation': row['max_correlation'],
                'significance_ratio': row['significant_count'] / row['measurement_count'],
                'industries': zone_data.get('industries', []),
                'monitoring_stations': zone_data.get('monitoring_stations', [])
            })
        
        return hotspot_list

def main():
    """Example usage"""
                             
    df = pd.read_parquet("data/processed_water_quality.parquet")
    
                                  
    tracker = SmartphoneLifecycleTracker()
    
                                         
    total_pollution = tracker.calculate_total_pollution_per_phone()
    print("Total pollution per smartphone:")
    for pollutant, amount in total_pollution.items():
        print(f"  {pollutant}: {amount:.3f}")
    
                                             
    analyzer = WaterQualityCorrelationAnalyzer(tracker)
    correlations = analyzer.analyze_temporal_correlations(df)
    
    print(f"\nFound {len(correlations)} correlations")
    significant_correlations = correlations[correlations['significant']]
    print(f"Significant correlations: {len(significant_correlations)}")
    
                                 
    hotspots = analyzer.identify_pollution_hotspots(df)
    print(f"\nPollution hotspots identified: {len(hotspots)}")
    for hotspot in hotspots[:3]:         
        print(f"  {hotspot['zone']}: avg_correlation={hotspot['avg_correlation']:.3f}")
    
                                             
    network_data = tracker.create_supply_chain_network_data()
    print(f"\nSupply chain network: {network_data['metadata']['total_nodes']} nodes, {network_data['metadata']['total_links']} links")
    
    print("Product lifecycle analysis completed")

if __name__ == "__main__":
    main()

