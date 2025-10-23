import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { Select } from './ui/Select';
import { Badge } from './ui/Badge';
import { MetricCard } from './ui/MetricCard';

interface ChoroplethMapData {
  map_type: string;
  geojson_data: any; // expect a valid GeoJSON FeatureCollection
  map_config: any;
  data_points: number;
  parameter: string;
  timestamp: string;
}

const ChinaMap: React.FC = () => {
  const [selectedParameter, setSelectedParameter] = useState<string>('ph');
  const [mapData, setMapData] = useState<ChoroplethMapData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // containerRef is used for responsive width measurement
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const parameters = useMemo(
    () => [
      { value: 'ph', label: 'pH值' },
      { value: 'dissolved_oxygen', label: '溶解氧' },
      { value: 'ammonia_nitrogen', label: '氨氮' },
      { value: 'total_phosphorus', label: '总磷' },
    ],
    []
  );

  const fetchMapData = async (parameter: string) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/api/map?parameter=${parameter}`);
      if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
      const result = await resp.json();
      if (result?.success && result?.data) {
        setMapData(result.data as ChoroplethMapData);
      } else {
        setError(result?.message || 'Failed to fetch map data');
      }
    } catch (e: any) {
      setError(e?.message || 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Fetch when parameter changes
  useEffect(() => {
    fetchMapData(selectedParameter);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedParameter]);

  // Helper: safe number formatting
  const safeNumber = (v: any) => (Number.isFinite(v) ? Number(v) : null);
  const fmt = (v: any, digits = 3) => {
    const n = safeNumber(v);
    return n === null ? '无数据' : n.toFixed(digits);
  };

  // Render map whenever data changes
  useEffect(() => {
    if (!mapData || !svgRef.current || !containerRef.current) return;

    const svgEl = svgRef.current;
    const containerEl = containerRef.current;

    // Clear previous render
    svgEl.innerHTML = '';

    // Fixed dimensions to display complete China map
    const width = 1000;
    const height = 800;

    // Set SVG size & viewBox to be responsive
    svgEl.setAttribute('width', String(width));
    svgEl.setAttribute('height', String(height));
    svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svgEl.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    const svg = d3.select(svgEl);

    // Remove old tooltip (if any), then create a new one
    d3.select('body').selectAll('div.map-tooltip').remove();
    const tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'map-tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '10px')
      .style('border-radius', '4px')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .style('font-size', '12px')
      .style('z-index', '1000');

    // Create a projection; Mercator is sufficient for China outline
    const projection = d3
      .geoMercator()
      .scale(width * 0.8) // Adjusted scale for complete China display
      .center([104.1954, 35.8617])
      .translate([width / 2, height / 2]);

    const path = d3.geoPath().projection(projection);

    // Layer group for pan/zoom
    const g = svg.append('g').attr('class', 'map-layer');

    // Draw features
    const features = mapData?.geojson_data?.features ?? [];

    g.selectAll('path.region')
      .data(features)
      .enter()
      .append('path')
      .attr('class', 'region')
      .attr('d', path as any)
      .attr('fill', (d: any) => d?.properties?.color || '#CCCCCC')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('mouseover', function (event: MouseEvent, d: any) {
        tooltip.transition().duration(150).style('opacity', 0.95);
        tooltip
          .html(
            `
            <strong>${d?.properties?.name ?? '未知区域'}</strong><br/>
            参数: ${mapData.parameter}<br/>
            数值: ${fmt(d?.properties?.value)}<br/>
            等级: ${d?.properties?.pollution_level ?? '—'}
          `
          )
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 28 + 'px');
      })
      .on('mousemove', function (event: MouseEvent) {
        tooltip.style('left', event.pageX + 10 + 'px').style('top', event.pageY - 28 + 'px');
      })
      .on('mouseout', function () {
        tooltip.transition().duration(200).style('opacity', 0);
      });

    // Zoom & pan
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 8])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Cleanup on unmount or data change
    return () => {
      tooltip.remove();
      svg.selectAll('*').remove();
    };
  }, [mapData]);

  const generateMap = () => {
    fetchMapData(selectedParameter);
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">中国水质污染轮廓图可视化</h1>

      <Card className="mb-6 p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">地图配置</h2>
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">选择监测参数</label>
            <Select
              value={selectedParameter}
              onChange={(value: string) => setSelectedParameter(value)}
              options={parameters}
            />
          </div>

          <div className="flex items-end space-x-2">
            <Button onClick={generateMap} disabled={loading}>
              {loading ? '生成中...' : '生成轮廓图'}
            </Button>
          </div>
        </div>

        {error && (
          <Badge variant="error" className="mb-4">
            错误: {error}
          </Badge>
        )}
      </Card>

      {mapData && (
        <>
          <Card className="p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              {mapData.parameter.toUpperCase()} 污染分布轮廓图
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <MetricCard
                title="数据点数量"
                value={String(mapData.data_points ?? 0)}
                unit="个"
                description="用于生成地图的数据点总数"
              />
              <MetricCard
                title="最新更新时间"
                value={
                  mapData.timestamp
                    ? new Date(mapData.timestamp).toLocaleString()
                    : '—'
                }
                description="地图数据最后更新的时间"
              />
              <MetricCard
                title="地图类型"
                value={mapData.map_type ?? 'Choropleth'}
                description="轮廓图（Choropleth Map）"
              />
            </div>

            {/* Responsive SVG container */}
            <div ref={containerRef} className="w-full flex justify-center">
              <svg
                ref={svgRef}
                className="border border-gray-300 rounded-lg shadow-lg"
                style={{ maxWidth: '100%' }}
              />
            </div>
          </Card>

          {/* Legend */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">污染等级图例</h3>
            <div className="flex flex-wrap justify-center gap-4">
              <div className="flex items-center">
                <div className="w-6 h-6 bg-green-600 mr-2 rounded border" />
                <span className="text-sm">优秀</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-green-300 mr-2 rounded border" />
                <span className="text-sm">良好</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-yellow-400 mr-2 rounded border" />
                <span className="text-sm">轻度污染</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-orange-500 mr-2 rounded border" />
                <span className="text-sm">中度污染</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-red-500 mr-2 rounded border" />
                <span className="text-sm">重度污染</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-red-800 mr-2 rounded border" />
                <span className="text-sm">严重污染</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-gray-400 mr-2 rounded border" />
                <span className="text-sm">无数据</span>
              </div>
            </div>
          </Card>
        </>
      )}

      {loading && (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          <span className="ml-2 text-gray-600">正在生成地图...</span>
        </div>
      )}
    </div>
  );
};

export default ChinaMap;
