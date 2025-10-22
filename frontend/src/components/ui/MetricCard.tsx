import React from 'react';
import { Card } from './Card';
import { Badge } from './Badge';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  trend?: {
    value: number;
    label: string;
    direction: 'up' | 'down' | 'neutral';
  };
  status?: 'good' | 'warning' | 'critical';
  description?: string;
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  trend,
  status,
  description,
  className = ''
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'good': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-900';
    }
  };

  const getTrendIcon = () => {
    if (!trend) return null;
    
    switch (trend.direction) {
      case 'up':
        return (
          <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 17l9.2-9.2M17 17V7H7" />
          </svg>
        );
      case 'down':
        return (
          <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 7l-9.2 9.2M7 7v10h10" />
          </svg>
        );
      default:
        return (
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
          </svg>
        );
    }
  };

  return (
    <Card className={`${className}`} variant="default" padding="md">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium text-gray-600">{title}</h4>
          {status && (
            <Badge 
              variant={
                status === 'good' ? 'success' : 
                status === 'warning' ? 'warning' : 
                'error'
              }
              size="sm"
            >
              {status === 'good' ? '正常' : status === 'warning' ? '警告' : '严重'}
            </Badge>
          )}
        </div>
        
        <div className="flex items-baseline space-x-2">
          <span className={`text-2xl font-bold ${getStatusColor()}`}>
            {typeof value === 'number' ? value.toFixed(2) : value}
          </span>
          {unit && (
            <span className="text-sm text-gray-500">{unit}</span>
          )}
        </div>
        
        {trend && (
          <div className="flex items-center space-x-1">
            {getTrendIcon()}
            <span className="text-sm text-gray-600">
              {trend.value > 0 ? '+' : ''}{trend.value}% {trend.label}
            </span>
          </div>
        )}
        
        {description && (
          <p className="text-xs text-gray-500">{description}</p>
        )}
      </div>
    </Card>
  );
};
