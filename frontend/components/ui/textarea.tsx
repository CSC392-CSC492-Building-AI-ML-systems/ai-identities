import { TextareaHTMLAttributes, forwardRef } from "react";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className = "", ...props }, ref) => (
    <textarea
      ref={ref}
      className={`w-full p-2 border rounded-md focus:ring focus:ring-blue-400 
                  text-gray-900 placeholder-gray-500 bg-white border-gray-300 
                  focus:border-blue-500 focus:outline-none resize-none ${className}`}
      {...props}
    />
  )
);

Textarea.displayName = "Textarea";
