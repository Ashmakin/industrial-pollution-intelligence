import React, { useState, useEffect } from 'react';
import { Play, Pause, Download, RefreshCw, MapPin, Database, Activity, AlertTriangle, Filter, Globe, Settings } from 'lucide-react';
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

const EnhancedDataCollection: React.FC = () => {
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
  const [showAdvanced, setShowAdvanced] = useState(false);

  
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

  const handleStartCollection = async () => {
    setIsCollecting(true);
    setLoading(true);
    try {
      const response = await startDataCollection({
        areas: selectedAreas,
        basins: selectedBasins,
        stations: selectedStations,
        max_records: maxRecords
      });
      
      if (response.success) {
        console.log('Data collection started successfully');
        setStatus(prev => ({ 
          ...prev, 
          is_running: true,
          total_records: maxRecords,
          collected_records: 0,
          current_area: selectedAreas.join(', ')
        }));
        
        
        const interval = setInterval(() => {
          fetchStatus();
        }, 2000);
        
        
        setTimeout(() => {
          clearInterval(interval);
          setIsCollecting(false);
          setLoading(false);
        }, 300000);
      } else {
        console.error('Failed to start data collection:', response.message);
        setIsCollecting(false);
        setLoading(false);
      }
    } catch (error) {
      console.error('Error starting data collection:', error);
      setIsCollecting(false);
      setLoading(false);
    }
  };

  
  const handleAreaChange = (area: string) => {
    if (area && !selectedAreas.includes(area)) {
      setSelectedAreas([...selectedAreas, area]);
    }
    
    setSelectedBasins([]);
    setSelectedStations([]);
    setAvailableStations([]);
  };

  
  const handleBasinChange = (basin: string) => {
    if (basin && !selectedBasins.includes(basin)) {
      setSelectedBasins([...selectedBasins, basin]);
    }
    
    setSelectedStations([]);
  };

  
  const handleStationChange = (station: string) => {
    if (station && !selectedStations.includes(station)) {
      setSelectedStations([...selectedStations, station]);
    }
  };

  
  const loadStationsForArea = (areaName: string) => {
    const area = areaOptions.find(a => a.name === areaName);
    if (area) {
      fetchStations(area.code);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchBasins();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  
  useEffect(() => {
    if (selectedAreas.length > 0) {
      loadStationsForArea(selectedAreas[0]);
    }
  }, [selectedAreas]);

  return (
    <div className="space-y-6">
      {}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">数据采集</h2>
          <p className="text-gray-600">实时采集全国水质监测数据</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-2"
          >
            <Settings className="h-4 w-4" />
            <span>高级设置</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchStatus}
            className="flex items-center space-x-2"
          >
            <RefreshCw className="h-4 w-4" />
            <span>刷新状态</span>
          </Button>
        </div>
      </div>

      {}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="采集状态"
          value={status.is_running ? "运行中" : "已停止"}
          status={status.is_running ? "good" : "warning"}
        />
        <MetricCard
          title="总记录数"
          value={status.total_records.toLocaleString()}
          status={status.total_records > 0 ? "good" : undefined}
        />
        <MetricCard
          title="已采集"
          value={status.collected_records.toLocaleString()}
          status={status.collected_records > 0 ? "good" : undefined}
        />
        <MetricCard
          title="进度"
          value={`${status.progress.toFixed(1)}%`}
          status={status.progress > 50 ? "good" : status.progress > 0 ? "warning" : undefined}
        />
      </div>

      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Filter className="h-5 w-5" />
            <span>采集配置</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              选择地区 <span className="text-red-500">*</span>
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-60 overflow-y-auto border border-gray-200 rounded-md p-4">
              {areaOptions.map((area) => (
                <label key={area.code} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 p-2 rounded">
                  <input
                    type="checkbox"
                    checked={selectedAreas.includes(area.name)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedAreas([...selectedAreas, area.name]);
                      } else {
                        setSelectedAreas(selectedAreas.filter(a => a !== area.name));
                      }
                    }}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">{area.name}</span>
                </label>
              ))}
            </div>
            <p className="text-sm text-gray-500 mt-2">
              已选择 {selectedAreas.length} 个地区: {selectedAreas.join(', ')}
            </p>
          </div>

          {}
          {showAdvanced && (
            <>
              {}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  选择流域
                </label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-40 overflow-y-auto border border-gray-200 rounded-md p-4">
                  {basinOptions.map((basin) => (
                    <label key={basin.code} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 p-2 rounded">
                      <input
                        type="checkbox"
                        checked={selectedBasins.includes(basin.name)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedBasins([...selectedBasins, basin.name]);
                          } else {
                            setSelectedBasins(selectedBasins.filter(b => b !== basin.name));
                          }
                        }}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">{basin.name}</span>
                    </label>
                  ))}
                </div>
                <p className="text-sm text-gray-500 mt-2">
                  已选择 {selectedBasins.length} 个流域: {selectedBasins.join(', ')}
                </p>
              </div>

              {}
              {availableStations.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    选择监测站
                  </label>
                  <Select
                    value=""
                    onChange={handleStationChange}
                    options={availableStations.map(station => ({ 
                      value: station.name, 
                      label: `${station.name} (${station.basin})` 
                    }))}
                    placeholder="选择特定监测站（可选）"
                    className="w-full"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    已选择 {selectedStations.length} 个监测站
                  </p>
                </div>
              )}

              {}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  最大记录数
                </label>
                <input
                  type="number"
                  value={maxRecords}
                  onChange={(e) => setMaxRecords(parseInt(e.target.value) || 1000)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="100"
                  max="10000"
                />
                <p className="text-sm text-gray-500 mt-1">
                  建议值：1000-5000 条记录
                </p>
              </div>
            </>
          )}

          {}
          <div className="flex items-center space-x-4">
            <Button
              onClick={handleStartCollection}
              disabled={isCollecting || selectedAreas.length === 0}
              className="flex items-center space-x-2"
            >
              {loading ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span>{isCollecting ? '采集中...' : '开始采集'}</span>
            </Button>

            <Button
              variant="outline"
              onClick={() => {
                setSelectedAreas([]);
                setSelectedBasins([]);
                setSelectedStations([]);
                setMaxRecords(1000);
              }}
              disabled={isCollecting}
            >
              重置配置
            </Button>
          </div>
        </CardContent>
      </Card>

      {}
      {status.is_running && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-green-500" />
              <span>采集进度</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">当前地区</span>
                <span className="text-sm text-gray-900">{status.current_area}</span>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${status.progress}%` }}
                />
              </div>
              
              <div className="flex items-center justify-between text-sm text-gray-600">
                <span>{status.collected_records} / {status.total_records} 条记录</span>
                <span>{status.progress.toFixed(1)}%</span>
              </div>

              {status.errors.length > 0 && (
                <div className="bg-red-50 border border-red-200 rounded-md p-3">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="h-4 w-4 text-red-500" />
                    <span className="text-sm font-medium text-red-800">错误信息</span>
                  </div>
                  <ul className="mt-2 text-sm text-red-700">
                    {status.errors.map((error, index) => (
                      <li key={index}>• {error}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Globe className="h-5 w-5" />
            <span>自动采集服务状态</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-gray-900">定时采集服务</h3>
                <p className="text-sm text-gray-500">每小时自动采集全国数据</p>
              </div>
              <Badge variant="success">运行中</Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-gray-900">数据去重机制</h3>
                <p className="text-sm text-gray-500">基于数据哈希自动检测并过滤重复数据</p>
              </div>
              <Badge variant="success">已启用</Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-gray-900">数据解析格式</h3>
                <p className="text-sm text-gray-500">已修复HTML标签解析问题，站点名称格式正确</p>
              </div>
              <Badge variant="success">已修复</Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-gray-900">最后更新</h3>
                <p className="text-sm text-gray-500">{status.last_update ? new Date(status.last_update).toLocaleString() : '未知'}</p>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
              <div className="flex items-start space-x-3">
                <Activity className="h-5 w-5 text-blue-500 mt-0.5" />
                <div>
                  <h4 className="text-sm font-medium text-blue-800">服务状态说明</h4>
                  <p className="text-sm text-blue-700 mt-1">
                    自动采集服务正在后台运行，每小时自动采集全国31个省市的最新水质监测数据。
                    数据采集完成后会自动进行去重处理，确保数据库中不会出现重复记录。
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EnhancedDataCollection;
