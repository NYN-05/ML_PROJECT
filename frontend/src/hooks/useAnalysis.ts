import { useState, useCallback } from 'react';
import { apiService } from '../services/api';

interface AnalysisResult {
  success: boolean;
  prediction: string;
  confidence: number;
  confidenceLabel: string;
  riskLevel: string;
  processingTime: string;
  timestamp: string;
  analysisId: string;
  method: string;
  isEnsemble: boolean;
  threshold: number;
  ensembleDetails?: {
    totalModels: number;
    realVoteWeight: number;
    fakeVoteWeight: number;
    realPercentage: number;
    agreementLevel: number;
    strategiesUsed: Record<string, string>;
    totalInferenceTimeMs: number;
  };
  individualModels?: Array<{
    modelName: string;
    prediction: string;
    confidence: number;
    weight: number;
    contributionPercentage: number;
    inferenceTimeMs: number;
  }>;
}

interface HistoryEntry {
  analysisId: string;
  prediction: string;
  confidence: number;
  timestamp: string;
  title: string;
  method?: string;
  isEnsemble?: boolean;
}

export function useAnalysis() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const analyze = useCallback(async (text: string, title = '') => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiService.analyze(text, title) as AnalysisResult;
      setResult(response);
      return response;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const response = await apiService.getHistory() as { history: HistoryEntry[] };
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