import React, { useState, useEffect } from 'react';
import { TrendingUp, Brain, Zap, Activity, AlertCircle, CheckCircle, MapPin, Filter } from 'lucide-react';
import { generateForecast, ForecastResult, getStations } from '../services/api';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { Select } from './ui/Select';
import { Badge } from './ui/Badge';
import { MetricCard } from './ui/MetricCard';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';


interface Station {
  name: string;
  basin: string;
  province: string;
}

const Forecasting: React.FC = () => {
  const [selectedStation, setSelectedStation] = useState('');
  const [selectedParameter, setSelectedParameter] = useState('ph');
  const [selectedHorizon, setSelectedHorizon] = useState('168');
  const [selectedModel, setSelectedModel] = useState<'lstm' | 'prophet' | 'ensemble'>('lstm');
  const [forecastResults, setForecastResults] = useState<ForecastResult[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [availableStations, setAvailableStations] = useState<Station[]>([]);
  const [selectedCity, setSelectedCity] = useState('');
  const [loading, setLoading] = useState(false);

  
  const cities = [
    '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
    '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
    '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
  ];

  const parameters = [
    { value: 'ph', label: 'pH值', unit: '', color: '#3B82F6' },
    { value: 'ammonia_nitrogen', label: '氨氮', unit: 'mg/L', color: '#EF4444' },
    { value: 'dissolved_oxygen', label: '溶解氧', unit: 'mg/L', color: '#10B981' },
    { value: 'total_phosphorus', label: '总磷', unit: 'mg/L', color: '#F59E0B' },
    { value: 'temperature', label: '温度', unit: '°C', color: '#8B5CF6' },
    { value: 'conductivity', label: '电导率', unit: 'μS/cm', color: '#06B6D4' },
  ];

  const horizons = [
    { value: '24', label: '24小时 (1天)' },
    { value: '72', label: '72小时 (3天)' },
    { value: '168', label: '168小时 (1周)' },
    { value: '720', label: '720小时 (1月)' },
  ];

  const models = [
    { 
      value: 'lstm', 
      label: 'LSTM神经网络', 
      description: '深度学习时间序列预测，适合复杂模式识别',
      icon: Brain
    },
    { 
      value: 'prophet', 
      label: 'Prophet模型', 
      description: 'Facebook开源预测模型，适合季节性数据',
      icon: TrendingUp
    },
    { 
      value: 'ensemble', 
      label: '集成模型', 
      description: '多模型融合预测，提高预测准确性',
      icon: Zap
    },
  ];


  
  const filteredStations = availableStations;

  
  useEffect(() => {
    if (selectedCity) {
      loadStationsForCity(selectedCity);
    }
  }, [selectedCity]);

  const loadStationsForCity = async (city: string) => {
    setLoading(true);
    try {
      
      const response = await fetch(`/api/pollution/stations?province=${encodeURIComponent(city)}`);
      console.log('API响应状态:', response.status);
      if (response.ok) {
        const data = await response.json();
        console.log('API响应数据:', data);
        if (data.success && data.data && data.data.length > 0) {
          
          const stations = data.data.map((station: any) => ({
            name: station.station_name,
            basin: station.watershed || '未知流域',
            province: station.province || city
          }));
          console.log('加载监测站成功:', stations);
          setAvailableStations(stations);
        } else {
          console.error('获取监测站数据失败:', data.error || '未知错误');
          setAvailableStations([]);
        }
      } else {
        console.error('获取监测站数据失败:', response.statusText);
        setAvailableStations([]);
      }
    } catch (error) {
      console.error('加载监测站失败:', error);
      setAvailableStations([]);
    } finally {
      setLoading(false);
    }
  };

  const generateNewForecast = async () => {
    if (!selectedStation) {
      alert('请先选择监测站点');
      return;
    }
    
    setIsGenerating(true);
    try {
      const result = await generateForecast({
        station: selectedStation,
        parameter: selectedParameter,
        horizon: parseInt(selectedHorizon),
        model: selectedModel,
      });

      if (result.success && result.data) {
        setForecastResults([result.data]);
      }
    } catch (error) {
      console.error('Forecast generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };


  const formatChartData = (result: ForecastResult) => {
    return result.predictions.map((pred, index) => ({
      time: new Date(pred.timestamp).toLocaleDateString(),
      predicted: pred.predicted_value,
      upper: pred.confidence_upper,
      lower: pred.confidence_lower,
      index
    }));
  };

  const getParameterInfo = (paramValue: string) => {
    return parameters.find(p => p.value === paramValue) || { label: paramValue, unit: '', color: '#3B82F6' };
  };

  const getModelInfo = (modelValue: string) => {
    return models.find(m => m.value === modelValue) || models[0];
  };

  const renderForecastChart = (result: ForecastResult) => {
    const data = formatChartData(result);
    const paramInfo = getParameterInfo(result.parameter);

    return (
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="time" 
              stroke="#666"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="#666"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)} ${paramInfo.unit}`,
                name === 'predicted' ? '预测值' : name === 'upper' ? '置信上限' : '置信下限'
              ]}
              labelFormatter={(label) => `时间: ${label}`}
            />
            <Legend />
            {result.predictions[0]?.confidence_upper && (
              <>
                <Line
                  type="monotone"
                  dataKey="upper"
                  stroke={paramInfo.color}
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  name="置信上限"
                />
                <Line
                  type="monotone"
                  dataKey="lower"
                  stroke={paramInfo.color}
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  name="置信下限"
                />
              </>
            )}
            <Line
              type="monotone"
              dataKey="predicted"
              stroke={paramInfo.color}
              strokeWidth={2}
              dot={{ fill: paramInfo.color, strokeWidth: 2, r: 4 }}
              name="预测值"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderForecastResult = (result: ForecastResult) => {
    const paramInfo = getParameterInfo(result.parameter);
    const modelInfo = getModelInfo(selectedModel);

    return (
      <Card variant="elevated" className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <modelInfo.icon className="h-5 w-5 text-blue-600" />
              <span>{result.station_name} - {paramInfo.label}</span>
            </div>
            <Badge variant="success">预测完成</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {}
            <div className="lg:col-span-2">
              <h4 className="text-sm font-medium text-gray-700 mb-3">预测趋势</h4>
              {renderForecastChart(result)}
            </div>
            
            {}
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-gray-700">模型性能</h4>
              <div className="space-y-3">
                <MetricCard
                  title="RMSE"
                  value={result.model_metrics?.rmse?.toFixed(3) || '0.000'}
                  unit={paramInfo.unit}
                  description="均方根误差"
                  status={result.model_metrics?.rmse && result.model_metrics.rmse < 0.5 ? 'good' : result.model_metrics?.rmse && result.model_metrics.rmse < 1.0 ? 'warning' : 'critical'}
                />
                <MetricCard
                  title="MAE"
                  value={result.model_metrics?.mae?.toFixed(3) || '0.000'}
                  unit={paramInfo.unit}
                  description="平均绝对误差"
                  status={result.model_metrics?.mae && result.model_metrics.mae < 0.3 ? 'good' : result.model_metrics?.mae && result.model_metrics.mae < 0.6 ? 'warning' : 'critical'}
                />
                <MetricCard
                  title="MAPE"
                  value={result.model_metrics?.mape?.toFixed(1) || '0.0'}
                  unit="%"
                  description="平均绝对百分比误差"
                  status={result.model_metrics?.mape && result.model_metrics.mape < 10 ? 'good' : result.model_metrics?.mape && result.model_metrics.mape < 20 ? 'warning' : 'critical'}
                />
              </div>
            </div>
          </div>

          {}
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-medium text-blue-900 mb-3 flex items-center">
              <Activity className="h-4 w-4 mr-2" />
              预测洞察
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-blue-700 font-medium">预测范围:</span>
                <span className="ml-2 text-blue-800">
                  {Math.min(...result.predictions.map(p => p.predicted_value)).toFixed(2)} - {Math.max(...result.predictions.map(p => p.predicted_value)).toFixed(2)} {paramInfo.unit}
                </span>
              </div>
              <div>
                <span className="text-blue-700 font-medium">预测点数:</span>
                <span className="ml-2 text-blue-800">{result.predictions.length}</span>
              </div>
              <div>
                <span className="text-blue-700 font-medium">数据点使用:</span>
                <span className="ml-2 text-blue-800">N/A</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 bg-purple-100 rounded-lg">
              <TrendingUp className="h-6 w-6 text-purple-600" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900">预测分析中心</h1>
          </div>
          <p className="text-lg text-gray-600">
            使用先进的机器学习模型预测水质参数变化趋势，为环境管理决策提供科学依据
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {}
          <div className="lg:col-span-1">
            <Card variant="elevated" className="sticky top-8">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-blue-600" />
                  <span>预测配置</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    选择城市
                  </label>
                  <Select
                    options={cities.map(c => ({ value: c, label: c }))}
                    value={selectedCity}
                    onChange={(value) => {
                      setSelectedCity(value);
                      setSelectedStation('');
                    }}
                    placeholder="选择城市"
                  />
                </div>

                {}
                {selectedCity && filteredStations.length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      选择监测站点
                    </label>
                    <Select
                      options={filteredStations.map(s => ({ 
                        value: s.name, 
                        label: s.name
                      }))}
                      value={selectedStation}
                      onChange={setSelectedStation}
                      placeholder="选择监测站点"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      共 {filteredStations.length} 个监测站点
                    </p>
                  </div>
                )}

                {}
                <Select
                  label="预测参数"
                  options={parameters.map(p => ({ value: p.value, label: p.label }))}
                  value={selectedParameter}
                  onChange={setSelectedParameter}
                />

                {}
                <Select
                  label="预测时长"
                  options={horizons}
                  value={selectedHorizon}
                  onChange={setSelectedHorizon}
                />

                {}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    预测模型
                  </label>
                  <div className="space-y-2">
                    {models.map((model) => {
                      const Icon = model.icon;
                      return (
                        <button
                          key={model.value}
                          onClick={() => setSelectedModel(model.value as 'lstm' | 'prophet' | 'ensemble')}
                          className={`w-full text-left p-3 rounded-lg border transition-all ${
                            selectedModel === model.value
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <div className="flex items-center space-x-2">
                            <Icon className="h-4 w-4" />
                            <span className="text-sm font-medium">{model.label}</span>
                          </div>
                          <p className="text-xs text-gray-500 mt-1">{model.description}</p>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {}
                <Button
                  onClick={generateNewForecast}
                  disabled={isGenerating}
                  isLoading={isGenerating}
                  leftIcon={<Zap className="h-4 w-4" />}
                  size="lg"
                  className="w-full"
                >
                  {isGenerating ? '生成中...' : '生成预测'}
                </Button>
              </CardContent>
            </Card>
          </div>

          {}
          <div className="lg:col-span-3">
            {forecastResults.length === 0 ? (
              <Card variant="outlined" className="text-center py-12">
                <TrendingUp className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">准备开始预测</h3>
                <p className="text-gray-500">
                  选择预测参数和模型，点击"生成预测"按钮进行趋势预测
                </p>
              </Card>
            ) : (
              <div className="space-y-6">
                {forecastResults.map((result, index) => (
                  <div key={index}>
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">
                          {result.station_name} - {getParameterInfo(result.parameter).label}
                        </h3>
                        <p className="text-sm text-gray-500">
                          刚刚
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="info">{getModelInfo(selectedModel).label}</Badge>
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      </div>
                    </div>
                    {renderForecastResult(result)}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Forecasting;