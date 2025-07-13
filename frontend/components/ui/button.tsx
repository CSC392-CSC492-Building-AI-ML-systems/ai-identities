import { ButtonHTMLAttributes } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  className?: string;
  variant?: 'default' | 'outline';
}

export function Button({ className = '', variant = 'default', ...props }: ButtonProps) {
  const baseStyles = 'px-4 py-2 rounded-lg transition-colors disabled:bg-gray-400';
  const variantStyles = {
    default: 'bg-blue-600 text-white hover:bg-blue-700',
    outline: 'border border-gray-300 text-gray-700 hover:bg-gray-100',
  };

  return (
    <button
      className={`${baseStyles} ${className} ${variantStyles[variant]}`}
      {...props}
    />
  );
}
