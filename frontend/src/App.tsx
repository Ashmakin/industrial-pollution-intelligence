import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Database, BarChart3, TrendingUp, FileText, Home, Activity } from 'lucide-react';
import DataCollection from './components/DataCollection';
import EnhancedDataCollection from './components/EnhancedDataCollection';
import DataAnalysis from './components/DataAnalysis';
import EnhancedDataAnalysis from './components/EnhancedDataAnalysis';
import Forecasting from './components/Forecasting';
import Reporting from './components/Reporting';
import ChinaMap from './components/ChinaMap';
import './App.css';

const Navigation: React.FC = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: '仪表盘', icon: Home },
    { path: '/data-collection', label: '数据采集', icon: Database },
    { path: '/analysis', label: '数据分析', icon: BarChart3 },
    { path: '/forecasting', label: '预测分析', icon: TrendingUp },
    { path: '/reports', label: '报表生成', icon: FileText },
    { path: '/map', label: '地图可视化', icon: Activity },
  ];

  return (
    <nav className="bg-white/95 backdrop-blur-md shadow-sm border-b border-gray-100 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <div className="p-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl shadow-sm">
                <Activity className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">污染智能分析系统</h1>
                <p className="text-xs text-gray-500 font-medium">Industrial Pollution Intelligence System</p>
              </div>
            </div>
            
            <div className="hidden md:flex space-x-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                      isActive
                        ? 'bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 shadow-sm border border-blue-100'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50/80'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 px-3 py-1.5 bg-gradient-to-r from-green-50 to-emerald-50 rounded-full border border-green-100">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs font-medium text-green-700">系统正常</span>
            </div>
            <div className="text-xs text-gray-500 font-medium bg-gray-50 px-3 py-1.5 rounded-lg">
              {new Date().toLocaleString()}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

const Dashboard: React.FC = () => {
  const [systemStatus, setSystemStatus] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    fetch('http://localhost:8080/health')
      .then(response => response.json())
      .then(data => {
        setSystemStatus(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error:', error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">加载系统状态中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {}
        <div className="mb-12">
          <div className="flex items-center space-x-4 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl shadow-lg">
              <Activity className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-gray-900">系统仪表盘</h1>
              <p className="text-lg text-gray-600 mt-1">
                实时监控工业污染智能分析系统的运行状态和关键指标
              </p>
            </div>
          </div>
        </div>

        {}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <div className="group bg-white/80 backdrop-blur-sm overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl shadow-sm group-hover:shadow-md transition-shadow">
                    <Activity className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-600">系统状态</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {systemStatus?.status === 'ok' ? '正常' : '异常'}
                    </p>
                  </div>
                </div>
                <div className={`w-3 h-3 rounded-full ${systemStatus?.status === 'ok' ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
              </div>
            </div>
            <div className="px-6 py-4 bg-gradient-to-r from-gray-50/80 to-gray-100/50 border-t border-gray-100/50">
              <p className="text-sm text-gray-600 font-medium">{systemStatus?.service || '未知服务'}</p>
            </div>
          </div>

          <div className="group bg-white/80 backdrop-blur-sm overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl shadow-sm group-hover:shadow-md transition-shadow">
                    <Database className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-600">后端服务</p>
                    <p className="text-2xl font-bold text-gray-900">在线</p>
                  </div>
                </div>
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </div>
            </div>
            <div className="px-6 py-4 bg-gradient-to-r from-gray-50/80 to-gray-100/50 border-t border-gray-100/50">
              <p className="text-sm text-gray-600 font-medium">Rust Axum API服务</p>
            </div>
          </div>

          <div className="group bg-white/80 backdrop-blur-sm overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-gradient-to-r from-purple-500 to-violet-500 rounded-xl shadow-sm group-hover:shadow-md transition-shadow">
                    <Database className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-600">数据库</p>
                    <p className="text-2xl font-bold text-gray-900">已连接</p>
                  </div>
                </div>
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </div>
            </div>
            <div className="px-6 py-4 bg-gradient-to-r from-gray-50/80 to-gray-100/50 border-t border-gray-100/50">
              <p className="text-sm text-gray-600 font-medium">PostgreSQL数据库</p>
            </div>
          </div>

          <div className="group bg-white/80 backdrop-blur-sm overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-gradient-to-r from-orange-500 to-red-500 rounded-xl shadow-sm group-hover:shadow-md transition-shadow">
                    <BarChart3 className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-600">数据记录</p>
                    <p className="text-2xl font-bold text-gray-900">1,200+</p>
                  </div>
                </div>
                <div className="w-3 h-3 bg-orange-500 rounded-full animate-pulse"></div>
              </div>
            </div>
            <div className="px-6 py-4 bg-gradient-to-r from-gray-50/80 to-gray-100/50 border-t border-gray-100/50">
              <p className="text-sm text-gray-600 font-medium">水质监测数据点</p>
            </div>
          </div>
        </div>

        {}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <div className="group bg-white/80 backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-3 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl shadow-sm">
                  <Database className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">数据采集模块</h3>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100/50">
                  <span className="text-sm font-medium text-gray-700">CNEMC API连接</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm font-semibold text-green-700">已连接</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">监测站点</span>
                  <span className="text-lg font-bold text-gray-900">5个站点</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">采集频率</span>
                  <span className="text-lg font-bold text-gray-900">实时</span>
                </div>
                <div className="pt-4">
                  <Link
                    to="/data-collection"
                    className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:-translate-y-0.5"
                  >
                    <Database className="h-4 w-4 mr-2" />
                    查看详情
                  </Link>
                </div>
              </div>
            </div>
          </div>

          <div className="group bg-white/80 backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-3 bg-gradient-to-r from-green-500 to-teal-500 rounded-xl shadow-sm">
                  <BarChart3 className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">数据分析模块</h3>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100/50">
                  <span className="text-sm font-medium text-gray-700">分析算法</span>
                  <div className="flex space-x-2">
                    <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded-full">PCA</span>
                    <span className="px-3 py-1 bg-purple-100 text-purple-800 text-xs font-semibold rounded-full">异常检测</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-100/50">
                  <span className="text-sm font-medium text-gray-700">因果分析</span>
                  <span className="px-3 py-1 bg-purple-100 text-purple-800 text-xs font-semibold rounded-full">格兰杰因果性</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">可视化</span>
                  <span className="text-lg font-bold text-gray-900">雷达图, 热力图</span>
                </div>
                <div className="pt-4">
                  <Link
                    to="/analysis"
                    className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-green-600 to-teal-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:-translate-y-0.5"
                  >
                    <BarChart3 className="h-4 w-4 mr-2" />
                    查看详情
                  </Link>
                </div>
              </div>
            </div>
          </div>

          <div className="group bg-white/80 backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl shadow-sm">
                  <TrendingUp className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">预测分析模块</h3>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-100/50">
                  <span className="text-sm font-medium text-gray-700">预测模型</span>
                  <div className="flex space-x-2">
                    <span className="px-3 py-1 bg-purple-100 text-purple-800 text-xs font-semibold rounded-full">LSTM</span>
                    <span className="px-3 py-1 bg-pink-100 text-pink-800 text-xs font-semibold rounded-full">Prophet</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100/50">
                  <span className="text-sm font-medium text-gray-700">预测精度</span>
                  <span className="text-lg font-bold text-green-700">RMSE &lt; 0.5</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">预测范围</span>
                  <span className="text-lg font-bold text-gray-900">1天 - 1月</span>
                </div>
                <div className="pt-4">
                  <Link
                    to="/forecasting"
                    className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:-translate-y-0.5"
                  >
                    <TrendingUp className="h-4 w-4 mr-2" />
                    查看详情
                  </Link>
                </div>
              </div>
            </div>
          </div>

          <div className="group bg-white/80 backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-300 rounded-2xl border border-gray-100/50 hover:-translate-y-1">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-3 bg-gradient-to-r from-orange-500 to-red-500 rounded-xl shadow-sm">
                  <FileText className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">报表生成模块</h3>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-orange-50 to-red-50 rounded-xl border border-orange-100/50">
                  <span className="text-sm font-medium text-gray-700">报告格式</span>
                  <div className="flex space-x-2">
                    <span className="px-3 py-1 bg-red-100 text-red-800 text-xs font-semibold rounded-full">PDF</span>
                    <span className="px-3 py-1 bg-green-100 text-green-800 text-xs font-semibold rounded-full">Excel</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100/50">
                  <span className="text-sm font-medium text-gray-700">生成频率</span>
                  <span className="text-lg font-bold text-blue-700">按需生成</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">报告类型</span>
                  <span className="text-lg font-bold text-gray-900">综合分析报告</span>
                </div>
                <div className="pt-4">
                  <Link
                    to="/reports"
                    className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:-translate-y-0.5"
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    查看详情
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>

        {}
        <div className="bg-white/80 backdrop-blur-sm shadow-lg rounded-2xl border border-gray-100/50">
          <div className="p-8">
            <div className="flex items-center space-x-3 mb-8">
              <div className="p-3 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-xl shadow-sm">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900">快速操作</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <Link
                to="/data-collection"
                className="group flex flex-col items-center p-6 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl border border-blue-100/50 hover:border-blue-300 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <div className="p-4 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  <Database className="h-8 w-8 text-white" />
                </div>
                <span className="text-sm font-semibold text-gray-700">开始采集</span>
              </Link>
              <Link
                to="/analysis"
                className="group flex flex-col items-center p-6 bg-gradient-to-br from-green-50 to-teal-50 rounded-2xl border border-green-100/50 hover:border-green-300 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <div className="p-4 bg-gradient-to-r from-green-500 to-teal-500 rounded-2xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  <BarChart3 className="h-8 w-8 text-white" />
                </div>
                <span className="text-sm font-semibold text-gray-700">数据分析</span>
              </Link>
              <Link
                to="/forecasting"
                className="group flex flex-col items-center p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl border border-purple-100/50 hover:border-purple-300 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <div className="p-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  <TrendingUp className="h-8 w-8 text-white" />
                </div>
                <span className="text-sm font-semibold text-gray-700">趋势预测</span>
              </Link>
              <Link
                to="/reports"
                className="group flex flex-col items-center p-6 bg-gradient-to-br from-orange-50 to-red-50 rounded-2xl border border-orange-100/50 hover:border-orange-300 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <div className="p-4 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  <FileText className="h-8 w-8 text-white" />
                </div>
                <span className="text-sm font-semibold text-gray-700">生成报告</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data-collection" element={<EnhancedDataCollection />} />
          <Route path="/analysis" element={<EnhancedDataAnalysis />} />
          <Route path="/forecasting" element={<Forecasting />} />
          <Route path="/reports" element={<Reporting />} />
          <Route path="/map" element={<ChinaMap />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;