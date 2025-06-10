import { InputHTMLAttributes, forwardRef } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string; // Add label as an optional prop
}

export const Input = forwardRef<HTMLInputElement, InputProps>(({ label, className = '', ...props }, ref) => (
  <div className="flex flex-col space-y-1">
    {label && (
      <label htmlFor={props.id || props.name} className="text-sm font-medium text-gray-700">
        {label}
      </label>
    )}
    <input
      ref={ref}
      className={`w-full p-2 border rounded-md focus:ring focus:ring-blue-400 
                  text-gray-900 placeholder-gray-500 bg-white border-gray-300 
                  focus:border-blue-500 focus:outline-none ${className}`}
      {...props}
    />
  </div>
));

Input.displayName = 'Input';
