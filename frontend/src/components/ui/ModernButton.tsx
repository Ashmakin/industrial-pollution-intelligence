import React from 'react';

interface ModernButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'gradient' | 'glass';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const ModernButton: React.FC<ModernButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  className = '',
  disabled,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transform active:scale-95';
  
  const variantClasses = {
    primary: 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 focus:ring-blue-500 shadow-lg hover:shadow-xl shadow-blue-500/25',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-500 shadow-md hover:shadow-lg',
    ghost: 'text-gray-700 hover:bg-gray-100 focus:ring-gray-500',
    gradient: 'bg-gradient-to-r from-purple-600 via-pink-600 to-red-500 text-white hover:from-purple-700 hover:via-pink-700 hover:to-red-600 focus:ring-purple-500 shadow-lg hover:shadow-xl shadow-purple-500/25',
    glass: 'bg-white/20 backdrop-blur-md text-gray-800 border border-white/30 hover:bg-white/30 focus:ring-white/50 shadow-lg hover:shadow-xl'
  };
  
  const sizeClasses = {
    sm: 'px-4 py-2 text-sm rounded-xl',
    md: 'px-6 py-3 text-sm rounded-xl',
    lg: 'px-8 py-4 text-base rounded-2xl'
  };
  
  const classes = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`;
  
  return (
    <button className={classes} disabled={disabled || isLoading} {...props}>
      {isLoading ? (
        <>
          <svg className="animate-spin -ml-1 mr-3 h-5 w-5" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          加载中...
        </>
      ) : (
        <>
          {leftIcon && <span className="mr-3">{leftIcon}</span>}
          {children}
          {rightIcon && <span className="ml-3">{rightIcon}</span>}
        </>
      )}
    </button>
  );
};
