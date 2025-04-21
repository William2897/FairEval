import { useState, useEffect } from 'react';

/**
 * A custom hook that debounces a value by the specified delay.
 * This is useful for reducing API calls when filtering or searching.
 * 
 * @param value The value to be debounced
 * @param delay The delay in milliseconds (default: 500ms)
 * @returns The debounced value
 */
export function useDebounce<T>(value: T, delay = 500): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}
