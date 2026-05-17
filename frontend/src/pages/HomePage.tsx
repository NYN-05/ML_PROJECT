import { useState, useEffect } from 'react';
import { MainLayout } from '../components/layout';
import { AnalysisForm, AnalysisResult } from '../components/features';
import { useAnalysis } from '../hooks/useAnalysis';
import styles from './HomePage.module.css';

export function HomePage() {
  const { isLoading, error, result, analyze, loadHistory, clearResult } = useAnalysis();
  const [showResult, setShowResult] = useState(false);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (result) {
      setShowResult(true);
    }
  }, [result]);

  const handleSubmit = async (text, title) => {
    try {
      await analyze(text, title);
    } catch (err) {
      console.error('Analysis failed:', err);
    }
  };

  const handleReset = () => {
    clearResult();
    setShowResult(false);
  };

  return (
    <MainLayout>
      <div className={styles.page}>
        <div className={styles.hero}>
          <h1 className={styles.title}>
            Fake News <span className={styles.highlight}>Detector</span>
          </h1>
          <p className={styles.subtitle}>
            Advanced AI-powered detection using an ensemble of 7 models with weighted voting
          </p>
        </div>

        <div className={styles.content}>
          {error && (
            <div className={styles.error}>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error}
            </div>
          )}

          {!showResult ? (
            <AnalysisForm onSubmit={handleSubmit} isLoading={isLoading} />
          ) : (
            <AnalysisResult result={result} onReset={handleReset} />
          )}
        </div>

        <div className={styles.features}>
          <div className={styles.feature}>
            <div className={styles.featureIcon}>
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
            </div>
            <h3>7 Models</h3>
            <p>Ensemble of 7 trained LSTM models</p>
          </div>
          <div className={styles.feature}>
            <div className={styles.featureIcon}>
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            </div>
            <h3>Weighted Voting</h3>
            <p>Performance-based model weights</p>
          </div>
          <div className={styles.feature}>
            <div className={styles.featureIcon}>
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            </div>
            <h3>Fast Analysis</h3>
            <p>Real-time detection in seconds</p>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}

export default HomePage;