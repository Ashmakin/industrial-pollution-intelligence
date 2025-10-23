import React, { useState, useEffect } from 'react';
import { Play, Pause, Download, RefreshCw, MapPin, Database, Activity, AlertTriangle, Filter, Globe } from 'lucide-react';
import { startDataCollection, getCollectionStatus } from '../services/api';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { Select } from './ui/Select';
import { Badge } from './ui/Badge';
import { MetricCard } from './ui/MetricCard';

interface CollectionStatus {
  is_running: boolean;
  progress: number;
  total_records: number;
  collected_records: number;
  current_area: string;
  last_update: string;
  errors: string[];
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

const DataCollection: React.FC = () => {
  const [status, setStatus] = useState<CollectionStatus>({
    is_running: false,
    progress: 0,
    total_records: 0,
    collected_records: 0,
    current_area: '',
    last_update: '',
    errors: []
  });

  const [selectedAreas, setSelectedAreas] = useState<string[]>(['北京', '上海', '广东']);
  const [selectedBasins, setSelectedBasins] = useState<string[]>([]);
  const [selectedStations, setSelectedStations] = useState<string[]>([]);
  const [maxRecords, setMaxRecords] = useState<number>(1000);
  const [isCollecting, setIsCollecting] = useState(false);
  const [availableAreas, setAvailableAreas] = useState<Area[]>([]);
  const [availableBasins, setAvailableBasins] = useState<Basin[]>([]);
  const [availableStations, setAvailableStations] = useState<Station[]>([]);
  const [loading, setLoading] = useState(false);

  
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

  const fetchStatus = async () => {
    try {
      const response = await getCollectionStatus();
      if (response.success && response.data) {
        setStatus(response.data);
      }
    } catch (error) {
      console.error('Failed to fetch collection status:', error);
      setStatus({
        is_running: false,
        progress: 0,
        total_records: 0,
        collected_records: 0,
        current_area: '未知',
        last_update: new Date().toISOString(),
        errors: [`连接错误: ${error}`]
      });
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStartCollection = async () => {
    setIsCollecting(true);
    try {
      const response = await startDataCollection({
        areas: selectedAreas,
        max_records: maxRecords
      });
      
      if (response.success) {
        setStatus(prev => ({ 
          ...prev, 
          is_running: true,
          total_records: maxRecords,
          collected_records: 0,
          current_area: selectedAreas.map(id => {
            const areaNames: { [key: string]: string } = { '110000': '北京市', '310000': '上海市', '440000': '广东省', '120000': '天津市', '500000': '重庆市' };
            return areaNames[id] || `区域${id}`;
          }).join(', ')
        }));
        
        setTimeout(() => {
          setStatus(prev => ({ ...prev, is_running: false, collected_records: maxRecords }));
          fetchStatus();
        }, 3000);
      }
    } catch (error) {
      console.error('Failed to start data collection:', error);
      setStatus(prev => ({
        ...prev,
        errors: [...prev.errors, `Collection failed: ${error}`]
      }));
    } finally {
      setIsCollecting(false);
    }
  };

  const handleStopCollection = () => {
    setStatus(prev => ({ ...prev, is_running: false }));
  };

  const handleAreaToggle = (areaCode: string) => {
    setSelectedAreas(prev => 
      prev.includes(areaCode) 
        ? prev.filter(code => code !== areaCode)
        : [...prev, areaCode]
    );
  };

  const progressPercentage = status.total_records > 0 ? (status.collected_records / status.total_records) * 100 : 0;

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Database className="h-6 w-6 text-blue-600" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900">数据采集中心</h1>
          </div>
          <p className="text-lg text-gray-600">
            实时从CNEMC API收集最新的水质监测数据，支持多区域并行采集
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {}
          <div className="lg:col-span-2 space-y-6">
            {}
            <Card variant="elevated">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <MapPin className="h-5 w-5 text-blue-600" />
                  <span>采集配置</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    监测区域选择
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {areaOptions.map(area => (
                      <label key={area.code} className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={selectedAreas.includes(area.code)}
                          onChange={() => handleAreaToggle(area.code)}
                          disabled={status.is_running}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded disabled:opacity-50"
                        />
                        <span className="text-sm text-gray-700">{area.name}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      最大记录数
                    </label>
                    <input
                      type="number"
                      value={maxRecords}
                      onChange={(e) => setMaxRecords(Number(e.target.value))}
                      min="100"
                      max="10000"
                      disabled={status.is_running}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      数据源
                    </label>
                    <div className="flex items-center space-x-2 px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg">
                      <Database className="h-4 w-4 text-gray-500" />
                      <span className="text-sm text-gray-700">CNEMC 国家环境监测中心</span>
                    </div>
                  </div>
                </div>

                {}
                <div className="flex space-x-3">
                  {!status.is_running ? (
                    <Button
                      onClick={handleStartCollection}
                      disabled={isCollecting || selectedAreas.length === 0}
                      leftIcon={<Play className="h-4 w-4" />}
                      size="lg"
                    >
                      {isCollecting ? '启动中...' : '开始采集'}
                    </Button>
                  ) : (
                    <Button
                      onClick={handleStopCollection}
                      variant="danger"
                      leftIcon={<Pause className="h-4 w-4" />}
                      size="lg"
                    >
                      停止采集
                    </Button>
                  )}
                  
                  <Button
                    onClick={fetchStatus}
                    variant="outline"
                    leftIcon={<RefreshCw className="h-4 w-4" />}
                    size="lg"
                  >
                    刷新状态
                  </Button>
                </div>
              </CardContent>
            </Card>

            {}
            <Card variant="elevated">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-5 w-5 text-green-600" />
                  <span>采集状态</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {status.is_running ? (
                  <div className="space-y-4">
                    {}
                    <div>
                      <div className="flex justify-between text-sm text-gray-600 mb-2">
                        <span>采集进度</span>
                        <span>{progressPercentage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${progressPercentage}%` }}
                        />
                      </div>
                    </div>

                    {}
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">已收集记录:</span>
                        <span className="ml-2 font-medium">{status.collected_records?.toLocaleString()}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">总记录数:</span>
                        <span className="ml-2 font-medium">{status.total_records?.toLocaleString()}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">当前区域:</span>
                        <span className="ml-2 font-medium">{status.current_area}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">最后更新:</span>
                        <span className="ml-2 font-medium">
                          {status.last_update ? new Date(status.last_update).toLocaleTimeString() : '未知'}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">数据采集已停止</p>
                    <p className="text-sm text-gray-400 mt-1">
                      总计收集了 {status.collected_records?.toLocaleString() || '0'} 条记录
                    </p>
                  </div>
                )}

                {}
                {status.errors.length > 0 && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center space-x-2 mb-3">
                      <AlertTriangle className="h-4 w-4 text-red-600" />
                      <h4 className="text-sm font-medium text-red-800">错误日志</h4>
                    </div>
                    <div className="space-y-2">
                      {status.errors.map((error, index) => (
                        <div key={index} className="text-sm text-red-700 bg-red-100 p-2 rounded">
                          {error}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {}
          <div className="space-y-6">
            {}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">关键指标</h3>
              <div className="space-y-4">
                <MetricCard
                  title="总记录数"
                  value={status.total_records?.toLocaleString() || '0'}
                  status={status.total_records > 0 ? 'good' : undefined}
                />
                <MetricCard
                  title="已收集"
                  value={status.collected_records?.toLocaleString() || '0'}
                  status={status.collected_records > 0 ? 'good' : undefined}
                />
                <MetricCard
                  title="监测区域"
                  value={selectedAreas.length}
                  unit="个"
                  status={selectedAreas.length > 0 ? 'good' : 'warning'}
                />
                <MetricCard
                  title="采集状态"
                  value={status.is_running ? '运行中' : '已停止'}
                  status={status.is_running ? 'good' : 'warning'}
                />
              </div>
            </div>

            {}
            <Card variant="outlined">
              <CardHeader>
                <CardTitle className="text-base">系统信息</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">数据源:</span>
                  <Badge variant="info">CNEMC API</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">更新频率:</span>
                  <span className="text-gray-900">实时</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">数据格式:</span>
                  <span className="text-gray-900">JSON</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">连接状态:</span>
                  <Badge variant={status.is_running ? 'success' : 'default'}>
                    {status.is_running ? '已连接' : '未连接'}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataCollection;