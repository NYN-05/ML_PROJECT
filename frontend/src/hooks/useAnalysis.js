import { useState, useCallback } from 'react';
import { apiService } from '../services/api';

export function useAnalysis() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const analyze = useCallback(async (text, title = '') => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiService.analyze(text, title);
      setResult(response);
      return response;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const response = await apiService.getHistory();
      setHistory(response.history || []);
      return response.history;
    } catch (err) {
      console.error('Failed to load history:', err);
      return [];
    }
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return {
    isLoading,
    error,
    result,
    history,
    analyze,
    loadHistory,
    clearResult,
  };
}

export default useAnalysis;