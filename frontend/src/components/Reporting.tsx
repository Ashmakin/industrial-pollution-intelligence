import React, { useState, useEffect } from 'react';
import { FileText, Download, Eye, BarChart3, TrendingUp, AlertTriangle, MapPin, Filter, Calendar, Settings } from 'lucide-react';
import { generateReport, downloadReport, getStations } from '../services/api';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { Select } from './ui/Select';
import { Badge } from './ui/Badge';

interface Station {
  name: string;
  basin: string;
  province: string;
}

interface ReportConfig {
  type: 'summary' | 'detailed' | 'forecast';
  stations: string[];
  timeRange: {
    start: string;
    end: string;
  };
  includeCharts: boolean;
  includeForecasts: boolean;
}

interface ReportResult {
  id: string;
  type: string;
  title: string;
  generated_at: string;
  status: 'generating' | 'completed' | 'failed';
  download_url?: string;
}

const Reporting: React.FC = () => {
  const [reportConfig, setReportConfig] = useState<ReportConfig>({
    type: 'summary',
    stations: [],
    timeRange: {
      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      end: new Date().toISOString().split('T')[0]
    },
    includeCharts: true,
    includeForecasts: false
  });

  const [generatedReports, setGeneratedReports] = useState<ReportResult[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [availableStations, setAvailableStations] = useState<Station[]>([]);
  const [selectedProvince, setSelectedProvince] = useState('');
  const [selectedBasin, setSelectedBasin] = useState('');
  const [loading, setLoading] = useState(false);

  
  const provinces = [
    '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
    '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
    '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
  ];

  const reportTypes = [
    {
      value: 'summary',
      label: '摘要报告',
      description: '包含关键指标和趋势分析的综合报告',
      icon: FileText
    },
    {
      value: 'detailed',
      label: '详细报告',
      description: '包含完整数据分析和可视化图表的详细报告',
      icon: BarChart3
    },
    {
      value: 'forecast',
      label: '预测报告',
      description: '包含未来趋势预测和政策建议的预测报告',
      icon: TrendingUp
    }
  ];

  
  const fetchStations = async () => {
    setLoading(true);
    try {
      const response = await getStations();
      if (response.success && response.data) {
        
        const stations = response.data.map((station: any) => ({
          name: station.station_name,
          basin: station.watershed || '未知流域',
          province: station.province || '未知省份'
        }));
        setAvailableStations(stations);
      }
    } catch (error) {
      console.error('Failed to fetch stations:', error);
    } finally {
      setLoading(false);
    }
  };

  
  const filteredStations = availableStations.filter(station => {
    if (selectedProvince && station.province !== selectedProvince) return false;
    if (selectedBasin && station.basin !== selectedBasin) return false;
    return true;
  });

  const handleStationToggle = (station: string) => {
    setReportConfig(prev => ({
      ...prev,
      stations: prev.stations.includes(station)
        ? prev.stations.filter(s => s !== station)
        : [...prev.stations, station]
    }));
  };

  
  useEffect(() => {
    fetchStations();
  }, []);

  
  useEffect(() => {
    if (selectedProvince) {
      loadStationsForProvince(selectedProvince);
    }
  }, [selectedProvince]);

  
  const loadStationsForProvince = async (province: string) => {
    try {
      const response = await getStations(province);
      if (response.success && response.data && response.data.length > 0) {
        
        const stations = response.data.map((station: any) => ({
          name: station.station_name,
          basin: station.watershed || '未知流域',
          province: station.province || province
        }));
        setAvailableStations(stations);
        } else {
          setAvailableStations([]);
        }
    } catch (error) {
      console.error('加载监测站失败:', error);
      setAvailableStations([]);
    }
  };

  const generateNewReport = async () => {
    setIsGenerating(true);
    try {
      const response = await generateReport({
        report_type: reportConfig.type,
        stations: reportConfig.stations.length > 0 ? reportConfig.stations : undefined,
        time_range: reportConfig.timeRange
      });

      if (response.success && response.data) {
        setGeneratedReports((prev: ReportResult[]) => [...prev, response.data!]);
      }
    } catch (error) {
      console.error('Report generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadReportFile = async (reportId: string) => {
    try {
      const blob = await downloadReport(reportId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pollution-report-${reportId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 bg-blue-100 rounded-lg">
              <FileText className="h-6 w-6 text-blue-600" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900">报表生成中心</h1>
          </div>
          <p className="text-lg text-gray-600">
            生成专业的污染分析报告和政策建议
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {}
          <div className="lg:col-span-2 space-y-6">
            {}
            <Card variant="elevated">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  <span>报告类型</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {reportTypes.map(type => {
                    const IconComponent = type.icon;
                    return (
                      <div
                        key={type.value}
                        className={`p-4 border rounded-lg cursor-pointer transition-all ${
                          reportConfig.type === type.value
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => setReportConfig(prev => ({ ...prev, type: type.value as any }))}
                      >
                        <div className="flex items-center space-x-3">
                          <IconComponent className="h-5 w-5 text-blue-500" />
                          <div>
                            <h3 className="font-medium text-gray-900">{type.label}</h3>
                            <p className="text-sm text-gray-500">{type.description}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {}
            <Card variant="elevated">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <MapPin className="h-5 w-5 text-blue-600" />
                  <span>选择监测站点</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    选择省份
                  </label>
                  <Select
                    options={provinces.map(province => ({ value: province, label: province }))}
                    value={selectedProvince}
                    onChange={(value) => {
                      setSelectedProvince(value);
                      setSelectedBasin('');
                      setReportConfig(prev => ({ ...prev, stations: [] }));
                    }}
                    placeholder="全部省份"
                  />
                </div>

                {}
                {selectedProvince && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      选择流域
                    </label>
                    <Select
                      options={Array.from(new Set(filteredStations.map(s => s.basin))).map(basin => ({ value: basin, label: basin }))}
                      value={selectedBasin}
                      onChange={(value) => {
                        setSelectedBasin(value);
                        setReportConfig(prev => ({ ...prev, stations: [] }));
                      }}
                      placeholder="全部流域"
                    />
                  </div>
                )}

                {}
                {selectedProvince && availableStations.length > 0 && (
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <label className="block text-sm font-medium text-gray-700">
                        监测站点
                      </label>
                      <label className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={availableStations.length > 0 && reportConfig.stations.length === availableStations.length}
                          onChange={(e) => {
                            setReportConfig(prev => ({
                              ...prev,
                              stations: e.target.checked ? availableStations.map(s => s.name) : []
                            }));
                          }}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-600">全选</span>
                      </label>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-60 overflow-y-auto border border-gray-200 rounded-md p-4">
                      {availableStations.map(station => (
                        <label key={station.name} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 p-2 rounded">
                          <input
                            type="checkbox"
                            checked={reportConfig.stations.includes(station.name)}
                            onChange={() => handleStationToggle(station.name)}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <div className="flex-1">
                            <span className="text-sm text-gray-700 font-medium">{station.name}</span>
                            <span className="text-xs text-gray-500 ml-2">({station.basin})</span>
                          </div>
                        </label>
                      ))}
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      已选择 {reportConfig.stations.length} 个监测站点
                    </p>
                  </div>
                )}
                
                {selectedProvince && availableStations.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <MapPin className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>当前省份暂无监测站点数据。</p>
                  </div>
                )}
                
                {!selectedProvince && (
                  <div className="text-center py-8 text-gray-500">
                    <MapPin className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>请先选择省份</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {}
            <Card variant="elevated">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Calendar className="h-5 w-5 text-blue-600" />
                  <span>时间范围和选项</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-3">时间范围</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-600 mb-1">开始日期</label>
                      <input
                        type="date"
                        value={reportConfig.timeRange.start}
                        onChange={(e) => setReportConfig(prev => ({
                          ...prev,
                          timeRange: { ...prev.timeRange, start: e.target.value }
                        }))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-600 mb-1">结束日期</label>
                      <input
                        type="date"
                        value={reportConfig.timeRange.end}
                        onChange={(e) => setReportConfig(prev => ({
                          ...prev,
                          timeRange: { ...prev.timeRange, end: e.target.value }
                        }))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                </div>

                {}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-3">报告选项</h4>
                  <div className="space-y-3">
                    <label className="flex items-center space-x-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={reportConfig.includeCharts}
                        onChange={(e) => setReportConfig(prev => ({
                          ...prev,
                          includeCharts: e.target.checked
                        }))}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <div>
                        <span className="text-sm font-medium text-gray-700">包含图表和可视化</span>
                        <p className="text-xs text-gray-500">在报告中包含图表和可视化分析</p>
                      </div>
                    </label>
                    <label className="flex items-center space-x-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={reportConfig.includeForecasts}
                        onChange={(e) => setReportConfig(prev => ({
                          ...prev,
                          includeForecasts: e.target.checked
                        }))}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <div>
                        <span className="text-sm font-medium text-gray-700">包含预测分析</span>
                        <p className="text-xs text-gray-500">在报告中包含未来趋势预测</p>
                      </div>
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {}
            <div className="flex justify-center">
              <Button
                onClick={generateNewReport}
                disabled={isGenerating}
                className="w-full max-w-md"
              >
                {isGenerating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    生成中...
                  </>
                ) : (
                  <>
                    <FileText className="h-4 w-4 mr-2" />
                    生成报告
                  </>
                )}
              </Button>
            </div>
          </div>

          {}
          <div className="lg:col-span-1">
            <Card variant="elevated" className="sticky top-8">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="h-5 w-5 text-blue-600" />
                  <span>已生成的报告</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {generatedReports.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <FileText className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>暂无生成的报告</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {generatedReports.map(report => (
                      <div key={report.id} className="border rounded-lg p-4">
                        <div className="flex items-start justify-between mb-2">
                          <h4 className="font-medium text-gray-900">{report.title}</h4>
                          <Badge variant={report.status === 'completed' ? 'success' : report.status === 'generating' ? 'warning' : 'error'}>
                            {report.status === 'completed' ? '已完成' : report.status === 'generating' ? '生成中' : '失败'}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">类型: {reportTypes.find(t => t.value === report.type)?.label}</p>
                        <p className="text-sm text-gray-600 mb-3">生成时间: {new Date(report.generated_at).toLocaleString()}</p>
                        {report.status === 'completed' && (
                          <div className="flex space-x-2">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => downloadReportFile(report.id)}
                            >
                              <Download className="h-3 w-3 mr-1" />
                              下载
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                            >
                              <Eye className="h-3 w-3 mr-1" />
                              预览
                            </Button>
                          </div>
                        )}
                        {report.status === 'generating' && (
                          <div className="flex items-center space-x-2 text-sm text-gray-500">
                            <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-500"></div>
                            <span>生成中...</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {}
        <div className="mt-8">
          <Card variant="elevated">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="h-5 w-5 text-blue-600" />
                <span>报告模板预览</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {}
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                    <BarChart3 className="h-4 w-4 mr-2 text-blue-500" />
                    执行摘要
                  </h4>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li key="stations-count">监测站点数量: {reportConfig.stations.length}</li>
                    <li key="time-range">分析时间范围: {reportConfig.timeRange.start} 至 {reportConfig.timeRange.end}</li>
                    <li key="main-findings">主要发现: 水质参数变化趋势分析</li>
                    <li key="key-recommendations">关键建议: 基于数据的政策建议</li>
                  </ul>
                </div>

                {}
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                    <TrendingUp className="h-4 w-4 mr-2 text-green-500" />
                    数据分析
                  </h4>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li key="data-quality">数据质量评估: 完整性 95%</li>
                    <li key="trend-analysis">趋势分析: 季节性变化识别</li>
                    <li key="correlation">相关性分析: 多参数关联性</li>
                    <li key="anomaly">异常检测: 识别异常数据点</li>
                  </ul>
                </div>

                {}
                <div className="border rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                    <AlertTriangle className="h-4 w-4 mr-2 text-orange-500" />
                    预测分析
                  </h4>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li key="forecast-model">预测模型: LSTM神经网络</li>
                    <li key="forecast-period">预测周期: 未来30天</li>
                    <li key="confidence">置信度: 85%</li>
                    <li key="risk-assessment">风险评估: 中等风险</li>
                  </ul>
                </div>

                {}
                <div className="border rounded-lg p-4 md:col-span-2 lg:col-span-3">
                  <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                    <Filter className="h-4 w-4 mr-2 text-purple-500" />
                    政策建议
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium text-gray-800 mb-2">短期措施</h5>
                      <ul className="space-y-1 text-sm text-gray-600">
                        <li>• 加强监测频率</li>
                        <li>• 优化处理工艺</li>
                        <li>• 应急响应预案</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-gray-800 mb-2">长期规划</h5>
                      <ul className="space-y-1 text-sm text-gray-600">
                        <li>• 基础设施升级</li>
                        <li>• 技术标准更新</li>
                        <li>• 可持续发展策略</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Reporting;
