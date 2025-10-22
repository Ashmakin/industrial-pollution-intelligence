import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, Network, Brain, Zap, Filter, MapPin, Droplets } from 'lucide-react';
import { runAnalysis } from '../services/api';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { Select } from './ui/Select';
import { Badge } from './ui/Badge';
import { MetricCard } from './ui/MetricCard';
import { BarChart, Bar, XAxis, YAxis, Tooltip, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

interface AnalysisResult {
  id: string;
  analysis_type: string;
  station_name: string;
  parameters: string[];
  timestamp: string;
  insights: string[];
  metrics: any;
  visualizations: any;
}

interface Area {
  code: string;
  name: string;
}

interface Basin {
  name: string;
  code: string;
}

interface Station {
  name: string;
  basin: string;
  area_id: string;
}

const EnhancedDataAnalysis: React.FC = () => {
  const [analysisType, setAnalysisType] = useState<'pca' | 'granger' | 'anomaly' | 'correlation'>('pca');
  const [selectedArea, setSelectedArea] = useState<string>('北京');
  const [selectedBasin, setSelectedBasin] = useState<string>('');
  const [selectedStation, setSelectedStation] = useState<string>('');
  const [selectedParameters, setSelectedParameters] = useState<string[]>(['ph', 'ammonia_nitrogen']);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [availableAreas, setAvailableAreas] = useState<Area[]>([]);
  const [availableBasins, setAvailableBasins] = useState<Basin[]>([]);
  const [availableStations, setAvailableStations] = useState<Station[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // 完整的中国省市列表
  const areaOptions = [
    { code: '110000', name: '北京' },
    { code: '120000', name: '天津' },
    { code: '130000', name: '河北' },
    { code: '140000', name: '山西' },
    { code: '150000', name: '内蒙古' },
    { code: '210000', name: '辽宁' },
    { code: '220000', name: '吉林' },
    { code: '230000', name: '黑龙江' },
    { code: '310000', name: '上海' },
    { code: '320000', name: '江苏' },
    { code: '330000', name: '浙江' },
    { code: '340000', name: '安徽' },
    { code: '350000', name: '福建' },
    { code: '360000', name: '江西' },
    { code: '370000', name: '山东' },
    { code: '410000', name: '河南' },
    { code: '420000', name: '湖北' },
    { code: '430000', name: '湖南' },
    { code: '440000', name: '广东' },
    { code: '450000', name: '广西' },
    { code: '460000', name: '海南' },
    { code: '500000', name: '重庆' },
    { code: '510000', name: '四川' },
    { code: '520000', name: '贵州' },
    { code: '530000', name: '云南' },
    { code: '540000', name: '西藏' },
    { code: '610000', name: '陕西' },
    { code: '620000', name: '甘肃' },
    { code: '630000', name: '青海' },
    { code: '640000', name: '宁夏' },
    { code: '650000', name: '新疆' }
  ];

  // 流域选项
  const basinOptions = [
    { name: '海河流域', code: 'haihe' },
    { name: '黄河流域', code: 'yellow_river' },
    { name: '长江流域', code: 'yangtze' },
    { name: '珠江流域', code: 'pearl_river' },
    { name: '松花江流域', code: 'songhua' },
    { name: '辽河流域', code: 'liaohe' },
    { name: '淮河流域', code: 'huaihe' },
    { name: '太湖流域', code: 'taihu' },
    { name: '巢湖流域', code: 'chaohu' },
    { name: '滇池流域', code: 'dianchi' },
    { name: '其他', code: 'other' }
  ];

  const analysisTypes = [
    { value: 'pca', label: '主成分分析 (PCA)', icon: BarChart3, description: '降维分析，识别主要变化模式' },
    { value: 'granger', label: '格兰杰因果性', icon: TrendingUp, description: '分析变量间的因果关系' },
    { value: 'anomaly', label: '异常检测', icon: AlertTriangle, description: '识别异常数据点' },
    { value: 'correlation', label: '相关性分析', icon: Network, description: '分析参数间相关性' },
  ];

  const parameters = [
    { value: 'ph', label: 'pH值' },
    { value: 'ammonia_nitrogen', label: '氨氮' },
    { value: 'dissolved_oxygen', label: '溶解氧' },
    { value: 'total_phosphorus', label: '总磷' },
    { value: 'temperature', label: '温度' },
    { value: 'conductivity', label: '电导率' },
    { value: 'turbidity', label: '浊度' },
    { value: 'permanganate_index', label: '高锰酸盐指数' },
    { value: 'total_nitrogen', label: '总氮' },
    { value: 'chlorophyll_a', label: '叶绿素α' },
    { value: 'algae_density', label: '藻密度' }
  ];

  // 获取可用的流域列表
  const fetchBasins = async () => {
    try {
      const response = await fetch('/api/basins');
      if (response.ok) {
        const data = await response.json();
        setAvailableBasins(data);
      }
    } catch (error) {
      console.error('Failed to fetch basins:', error);
    }
  };

  // 获取指定地区的监测站列表
  const fetchStations = async (areaId: string) => {
    try {
      const response = await fetch(`/api/stations?area_id=${areaId}`);
      if (response.ok) {
        const data = await response.json();
        setAvailableStations(data);
      }
    } catch (error) {
      console.error('Failed to fetch stations:', error);
    }
  };

  // 处理地区选择变化
  const handleAreaChange = (area: string) => {
    setSelectedArea(area);
    setSelectedBasin('');
    setSelectedStation('');
    setAvailableStations([]);
    
    // 加载该地区的监测站
    const areaCode = areaOptions.find(a => a.name === area)?.code;
    if (areaCode) {
      fetchStations(areaCode);
    }
  };

  // 处理流域选择变化
  const handleBasinChange = (basin: string) => {
    setSelectedBasin(basin);
    setSelectedStation('');
  };

  // 处理监测站选择变化
  const handleStationChange = (station: string) => {
    setSelectedStation(station);
  };

  const handleAnalysis = async () => {
    if (selectedParameters.length === 0) {
      alert('请至少选择一个参数进行分析');
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await runAnalysis({
        analysis_type: analysisType,
        stations: selectedStation ? [selectedStation] : undefined,
        parameters: selectedParameters
      });

      if (response.success && response.data) {
        const newResult: AnalysisResult = {
          id: Date.now().toString(),
          analysis_type: analysisType,
          station_name: selectedStation || `${selectedArea}${selectedBasin ? ` - ${selectedBasin}` : ''}`,
          parameters: selectedParameters,
          timestamp: new Date().toISOString(),
          insights: response.data.insights || [],
          metrics: response.data.metrics || {},
          visualizations: response.data.visualizations || {}
        };

        setAnalysisResults(prev => [newResult, ...prev]);
      } else {
        alert('分析失败: ' + (response.message || '未知错误'));
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert('分析过程中发生错误');
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    fetchBasins();
    setAvailableAreas(areaOptions);
  }, []);

  // 当选择地区时，加载监测站
  useEffect(() => {
    if (selectedArea) {
      const areaCode = areaOptions.find(a => a.name === selectedArea)?.code;
      if (areaCode) {
        fetchStations(areaCode);
      }
    }
  }, [selectedArea]);

  return (
    <div className="space-y-6">
      {/* 标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">数据分析</h2>
          <p className="text-gray-600">智能分析水质监测数据</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center space-x-2"
        >
          <Filter className="h-4 w-4" />
          <span>{showAdvanced ? '隐藏' : '显示'}高级选项</span>
        </Button>
      </div>

      {/* 分析配置 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>分析配置</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* 分析类型 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              分析类型 <span className="text-red-500">*</span>
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {analysisTypes.map((type) => {
                const Icon = type.icon;
                return (
                  <div
                    key={type.value}
                    className={`p-4 border rounded-lg cursor-pointer transition-all ${
                      analysisType === type.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setAnalysisType(type.value as any)}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon className="h-5 w-5 text-blue-500" />
                      <div>
                        <h3 className="font-medium text-gray-900">{type.label}</h3>
                        <p className="text-sm text-gray-500">{type.description}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 地区选择 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              选择地区 <span className="text-red-500">*</span>
            </label>
            <Select
              value={selectedArea}
              onChange={handleAreaChange}
              options={areaOptions.map(area => ({ value: area.name, label: area.name }))}
              placeholder="选择分析地区"
              className="w-full"
            />
          </div>

          {/* 高级选项 */}
          {showAdvanced && (
            <>
              {/* 流域选择 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  选择流域
                </label>
                <Select
                  value={selectedBasin}
                  onChange={handleBasinChange}
                  options={basinOptions.map(basin => ({ value: basin.name, label: basin.name }))}
                  placeholder="选择特定流域（可选）"
                  className="w-full"
                />
              </div>

              {/* 监测站选择 */}
              {availableStations.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    选择监测站
                  </label>
                  <Select
                    value={selectedStation}
                    onChange={handleStationChange}
                    options={availableStations.map(station => ({ 
                      value: station.name, 
                      label: `${station.name} (${station.basin})` 
                    }))}
                    placeholder="选择特定监测站（可选）"
                    className="w-full"
                  />
                </div>
              )}
            </>
          )}

          {/* 参数选择 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              选择参数 <span className="text-red-500">*</span>
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {parameters.map(param => (
                <label key={param.value} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedParameters.includes(param.value)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedParameters([...selectedParameters, param.value]);
                      } else {
                        setSelectedParameters(selectedParameters.filter(p => p !== param.value));
                      }
                    }}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="text-sm text-gray-700">{param.label}</span>
                </label>
              ))}
            </div>
            <p className="text-sm text-gray-500 mt-1">
              已选择 {selectedParameters.length} 个参数
            </p>
          </div>

          {/* 分析按钮 */}
          <Button
            onClick={handleAnalysis}
            disabled={isAnalyzing || selectedParameters.length === 0}
            className="w-full flex items-center justify-center space-x-2"
          >
            {isAnalyzing ? (
              <Zap className="h-4 w-4 animate-spin" />
            ) : (
              <Brain className="h-4 w-4" />
            )}
            <span>{isAnalyzing ? '分析中...' : '开始分析'}</span>
          </Button>
        </CardContent>
      </Card>

      {/* 分析结果 */}
      {analysisResults.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">分析结果</h3>
          {analysisResults.map((result) => (
            <Card key={result.id}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Badge variant="info">{result.analysis_type.toUpperCase()}</Badge>
                    <span>{result.station_name}</span>
                  </div>
                  <span className="text-sm text-gray-500">
                    {new Date(result.timestamp).toLocaleString()}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* 参数信息 */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">分析参数</h4>
                    <div className="flex flex-wrap gap-2">
                      {result.parameters.map((param) => (
                        <Badge key={param} variant="default">
                          {parameters.find(p => p.value === param)?.label || param}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* 洞察结果 */}
                  {result.insights.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">分析洞察</h4>
                      <ul className="space-y-1">
                        {result.insights.map((insight, index) => (
                          <li key={index} className="text-sm text-gray-600 flex items-start space-x-2">
                            <span className="text-blue-500 mt-1">•</span>
                            <span>{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* 可视化图表 */}
                  {result.visualizations && Object.keys(result.visualizations).length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">可视化结果</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(result.visualizations).map(([key, data]) => {
                          const chartData = data as any;
                          return (
                            <div key={key} className="border rounded-lg p-4">
                              <h5 className="text-sm font-medium text-gray-700 mb-2">{key}</h5>
                              {key === 'radar_chart' && chartData && Array.isArray(chartData) && (
                                <ResponsiveContainer width="100%" height={200}>
                                  <RadarChart data={chartData}>
                                    <PolarGrid />
                                    <PolarAngleAxis dataKey="parameter" />
                                    <PolarRadiusAxis />
                                    <Radar
                                      name="value"
                                      dataKey="value"
                                      stroke="#8884d8"
                                      fill="#8884d8"
                                      fillOpacity={0.6}
                                    />
                                  </RadarChart>
                                </ResponsiveContainer>
                              )}
                              {key === 'bar_chart' && chartData && Array.isArray(chartData) && (
                                <ResponsiveContainer width="100%" height={200}>
                                  <BarChart data={chartData}>
                                    <XAxis dataKey="name" />
                                    <YAxis />
                                    <Tooltip />
                                    <Bar dataKey="value" fill="#8884d8" />
                                  </BarChart>
                                </ResponsiveContainer>
                              )}
                              {key === 'scatter_data' && chartData && Array.isArray(chartData) && (
                                <div className="space-y-2">
                                  {chartData.slice(0, 5).map((item: any, index: number) => (
                                    <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                                      <div>
                                        <span className="font-medium">{item.station}</span>
                                        <span className="text-gray-500 ml-2">({item.parameter})</span>
                                      </div>
                                      <div className="text-right">
                                        <div className="text-sm font-medium">{item.value.toFixed(2)}</div>
                                        <div className={`text-xs ${item.severity === 'high' ? 'text-red-600' : 'text-yellow-600'}`}>
                                          Z-score: {item.z_score.toFixed(2)}
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                  {chartData.length > 5 && (
                                    <div className="text-center text-sm text-gray-500">
                                      还有 {chartData.length - 5} 个异常点...
                                    </div>
                                  )}
                                </div>
                              )}
                              {key === 'heatmap_data' && chartData && Array.isArray(chartData) && (
                                <div className="text-sm text-gray-600">
                                  <p>相关性矩阵数据已生成，包含 {chartData.length} 个参数对的相关性信息。</p>
                                  <p className="mt-2">数据格式：参数对 → 相关系数</p>
                                </div>
                              )}
                              {key === 'network_data' && chartData && Array.isArray(chartData) && (
                                <div className="space-y-2">
                                  {chartData.slice(0, 5).map((item: any, index: number) => (
                                    <div key={index} className="flex items-center p-2 bg-gray-50 rounded">
                                      <span className="font-medium">{item.source}</span>
                                      <span className="mx-2 text-gray-400">→</span>
                                      <span className="font-medium">{item.target}</span>
                                      <span className="ml-auto text-sm text-gray-600">
                                        强度: {(item.strength).toFixed(1)}%
                                      </span>
                                    </div>
                                  ))}
                                  {chartData.length > 5 && (
                                    <div className="text-center text-sm text-gray-500">
                                      还有 {chartData.length - 5} 个因果关系...
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default EnhancedDataAnalysis;
