import React, { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface AnimatedMetricCardProps {
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
  animated?: boolean;
}

export const AnimatedMetricCard: React.FC<AnimatedMetricCardProps> = ({
  title,
  value,
  unit,
  trend,
  status,
  description,
  className = '',
  animated = true
}) => {
  const [displayValue, setDisplayValue] = useState(animated ? 0 : value);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    const element = document.getElementById(`metric-${title}`);
    if (element) {
      observer.observe(element);
    }

    return () => observer.disconnect();
  }, [title]);

  useEffect(() => {
    if (isVisible && animated && typeof value === 'number') {
      const duration = 2000;
      const steps = 60;
      const increment = value / steps;
      const stepDuration = duration / steps;

      let current = 0;
      const timer = setInterval(() => {
        current += increment;
        if (current >= value) {
          setDisplayValue(value);
          clearInterval(timer);
        } else {
          setDisplayValue(Math.floor(current));
        }
      }, stepDuration);

      return () => clearInterval(timer);
    }
  }, [isVisible, value, animated]);

  const getStatusColor = () => {
    switch (status) {
      case 'good': return 'from-green-500 to-emerald-500';
      case 'warning': return 'from-yellow-500 to-orange-500';
      case 'critical': return 'from-red-500 to-pink-500';
      default: return 'from-gray-600 to-gray-700';
    }
  };

  const getTrendIcon = () => {
    if (!trend) return null;
    
    switch (trend.direction) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div
      id={`metric-${title}`}
      className={`group relative overflow-hidden rounded-2xl bg-gradient-to-br from-white via-blue-50/30 to-indigo-50/30 border border-white/20 shadow-xl hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 ${className}`}
    >
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-pink-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      
      <div className="relative p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-medium text-gray-600">{title}</h4>
          {status && (
            <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${getStatusColor()}`} />
          )}
        </div>
        
        <div className="flex items-baseline space-x-2 mb-2">
          <span className={`text-3xl font-bold bg-gradient-to-r ${getStatusColor()} bg-clip-text text-transparent`}>
            {typeof displayValue === 'number' && animated ? displayValue.toLocaleString() : value}
          </span>
          {unit && (
            <span className="text-sm text-gray-500">{unit}</span>
          )}
        </div>
        
        {trend && (
          <div className="flex items-center space-x-2">
            {getTrendIcon()}
            <span className="text-sm text-gray-600">
              {trend.value > 0 ? '+' : ''}{trend.value}% {trend.label}
            </span>
          </div>
        )}
        
        {description && (
          <p className="text-xs text-gray-500 mt-2">{description}</p>
        )}
      </div>
    </div>
  );
};
