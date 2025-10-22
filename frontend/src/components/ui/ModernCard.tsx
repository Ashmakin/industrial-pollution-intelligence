import React from 'react';

interface ModernCardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'glass' | 'gradient' | 'elevated' | 'minimal';
  padding?: 'sm' | 'md' | 'lg';
  hover?: boolean;
}

export const ModernCard: React.FC<ModernCardProps> = ({ 
  children, 
  className = '', 
  variant = 'glass',
  padding = 'md',
  hover = true
}) => {
  const baseClasses = 'rounded-2xl border transition-all duration-300';
  
  const variantClasses = {
    glass: 'bg-white/80 backdrop-blur-xl border-white/20 shadow-xl shadow-gray-200/50',
    gradient: 'bg-gradient-to-br from-white via-blue-50/50 to-indigo-50/50 border-white/30 shadow-xl shadow-blue-200/20',
    elevated: 'bg-white border-gray-200/50 shadow-2xl shadow-gray-200/30',
    minimal: 'bg-white/60 backdrop-blur-sm border-gray-100/50 shadow-lg shadow-gray-100/30'
  };
  
  const paddingClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };
  
  const hoverClasses = hover ? 'hover:shadow-2xl hover:shadow-gray-300/40 hover:-translate-y-1' : '';
  
  return (
    <div className={`${baseClasses} ${variantClasses[variant]} ${paddingClasses[padding]} ${hoverClasses} ${className}`}>
      {children}
    </div>
  );
};

export const ModernCardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <div className={`mb-6 ${className}`}>
    {children}
  </div>
);

export const ModernCardTitle: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <h3 className={`text-xl font-bold bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 bg-clip-text text-transparent ${className}`}>
    {children}
  </h3>
);

export const ModernCardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <div className={className}>
    {children}
  </div>
);
