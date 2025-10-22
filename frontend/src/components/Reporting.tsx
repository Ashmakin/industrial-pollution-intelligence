import React, { useState } from 'react';
import { FileText, Download, Eye, BarChart3, TrendingUp, AlertTriangle } from 'lucide-react';
import { generateReport, downloadReport } from '../services/api';

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

  const stations = [
    'Beijing Station', 'Shanghai Station', 'Guangdong Station', 
    'Tianjin Station', 'Chongqing Station'
  ];

  const reportTypes = [
    {
      value: 'summary',
      label: '摘要报告',
      description: '包含关键指标和趋势分析的综合报告',
      icon: <FileText size={20} />
    },
    {
      value: 'detailed',
      label: '详细报告',
      description: '包含完整数据分析和可视化图表的详细报告',
      icon: <BarChart3 size={20} />
    },
    {
      value: 'forecast',
      label: '预测报告',
      description: '包含未来趋势预测和政策建议的预测报告',
      icon: <TrendingUp size={20} />
    }
  ];

  const handleStationToggle = (station: string) => {
    setReportConfig(prev => ({
      ...prev,
      stations: prev.stations.includes(station)
        ? prev.stations.filter(s => s !== station)
        : [...prev.stations, station]
    }));
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
    <div className="reporting">
      <div className="section-header">
        <h2>📋 报表生成</h2>
        <p>生成专业的污染分析报告和政策建议</p>
      </div>

      <div className="report-config">
        <div className="config-section">
          <h3>报告类型</h3>
          <div className="report-types">
            {reportTypes.map(type => (
              <label key={type.value} className="report-type-option">
                <input
                  type="radio"
                  name="report-type"
                  value={type.value}
                  checked={reportConfig.type === type.value}
                  onChange={(e) => setReportConfig(prev => ({ ...prev, type: e.target.value as any }))}
                />
                <div className="type-content">
                  <div className="type-icon">{type.icon}</div>
                  <div className="type-info">
                    <strong>{type.label}</strong>
                    <p>{type.description}</p>
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        <div className="config-section">
          <h3>选择监测站点</h3>
          <div className="station-selection">
            <div className="select-all">
              <label>
                <input
                  type="checkbox"
                  checked={reportConfig.stations.length === stations.length}
                  onChange={(e) => {
                    setReportConfig(prev => ({
                      ...prev,
                      stations: e.target.checked ? [...stations] : []
                    }));
                  }}
                />
                全选
              </label>
            </div>
            <div className="station-grid">
              {stations.map(station => (
                <label key={station} className="station-option">
                  <input
                    type="checkbox"
                    checked={reportConfig.stations.includes(station)}
                    onChange={() => handleStationToggle(station)}
                  />
                  <span>{station}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="config-section">
          <h3>时间范围</h3>
          <div className="time-range">
            <div className="date-inputs">
              <label>
                开始日期:
                <input
                  type="date"
                  value={reportConfig.timeRange.start}
                  onChange={(e) => setReportConfig(prev => ({
                    ...prev,
                    timeRange: { ...prev.timeRange, start: e.target.value }
                  }))}
                />
              </label>
              <label>
                结束日期:
                <input
                  type="date"
                  value={reportConfig.timeRange.end}
                  onChange={(e) => setReportConfig(prev => ({
                    ...prev,
                    timeRange: { ...prev.timeRange, end: e.target.value }
                  }))}
                />
              </label>
            </div>
          </div>
        </div>

        <div className="config-section">
          <h3>报告选项</h3>
          <div className="report-options">
            <label className="option-item">
              <input
                type="checkbox"
                checked={reportConfig.includeCharts}
                onChange={(e) => setReportConfig(prev => ({ ...prev, includeCharts: e.target.checked }))}
              />
              包含图表和可视化
            </label>
            <label className="option-item">
              <input
                type="checkbox"
                checked={reportConfig.includeForecasts}
                onChange={(e) => setReportConfig(prev => ({ ...prev, includeForecasts: e.target.checked }))}
              />
              包含预测分析
            </label>
          </div>
        </div>

        <div className="action-buttons">
          <button
            onClick={generateNewReport}
            disabled={isGenerating}
            className="btn-primary"
          >
            <FileText size={16} />
            {isGenerating ? '生成中...' : '生成报告'}
          </button>
        </div>
      </div>

      <div className="generated-reports">
        <h3>已生成的报告</h3>
        {generatedReports.length === 0 ? (
          <div className="no-reports">
            <p>暂无生成的报告</p>
          </div>
        ) : (
          <div className="reports-list">
            {generatedReports.map(report => (
              <div key={report.id} className="report-item">
                <div className="report-info">
                  <h4>{report.title}</h4>
                  <p>类型: {reportTypes.find(t => t.value === report.type)?.label}</p>
                  <p>生成时间: {new Date(report.generated_at).toLocaleString()}</p>
                  <p>状态: 
                    <span className={`status ${report.status}`}>
                      {report.status === 'generating' ? '生成中' : 
                       report.status === 'completed' ? '已完成' : '失败'}
                    </span>
                  </p>
                </div>
                <div className="report-actions">
                  {report.status === 'completed' && (
                    <>
                      <button
                        onClick={() => downloadReportFile(report.id)}
                        className="btn-secondary"
                      >
                        <Download size={16} />
                        下载PDF
                      </button>
                      <button className="btn-secondary">
                        <Eye size={16} />
                        预览
                      </button>
                    </>
                  )}
                  {report.status === 'generating' && (
                    <div className="generating-indicator">
                      <div className="spinner"></div>
                      <span>生成中...</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="report-templates">
        <h3>报告模板预览</h3>
        <div className="template-preview">
          <div className="template-section">
            <h4>📊 执行摘要</h4>
            <ul>
              <li>监测站点数量: {reportConfig.stations.length || stations.length}</li>
              <li>分析时间范围: {reportConfig.timeRange.start} 至 {reportConfig.timeRange.end}</li>
              <li>主要发现: 水质参数变化趋势分析</li>
              <li>关键建议: 基于数据的政策建议</li>
            </ul>
          </div>

          <div className="template-section">
            <h4>📈 数据分析</h4>
            <ul>
              <li>统计概览: 各参数的平均值、最大值、最小值</li>
              <li>趋势分析: 时间序列变化趋势</li>
              <li>相关性分析: 污染物间的关系</li>
              <li>异常检测: 识别异常污染事件</li>
            </ul>
          </div>

          <div className="template-section">
            <h4>🔮 预测分析</h4>
            <ul>
              <li>未来趋势预测: 基于机器学习模型</li>
              <li>风险评估: 潜在污染风险</li>
              <li>模型性能: 预测准确性评估</li>
            </ul>
          </div>

          <div className="template-section">
            <h4>💡 政策建议</h4>
            <ul>
              <li>监测网络优化建议</li>
              <li>污染源控制策略</li>
              <li>预警系统部署建议</li>
              <li>长期管理规划</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Reporting;
