import { Card, Badge, Progress, Button } from '../ui';
import styles from './AnalysisResult.module.css';

export function AnalysisResult({ result, onReset }) {
  if (!result) return null;

  const isFake = result.prediction === 'FAKE';
  const confidence = result.confidence || 0;

  return (
    <div className={styles.result}>
      <Card className={styles.mainCard} padding="large">
        <div className={styles.header}>
          <div className={styles.verdictWrapper}>
            <div className={`${styles.icon} ${isFake ? styles.fake : styles.real}`}>
              {isFake ? (
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" />
                  <line x1="9" y1="9" x2="15" y2="15" />
                </svg>
              ) : (
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
              )}
            </div>
            <div className={styles.verdictText}>
              <span className={styles.label}>Prediction</span>
              <h2 className={`${styles.verdict} ${isFake ? styles.fake : styles.real}`}>
                {isFake ? 'Likely Fake' : 'Likely Real'}
              </h2>
            </div>
          </div>

          <Badge
            variant={result.confidenceLabel === 'HIGH' ? 'success' : result.confidenceLabel === 'MEDIUM' ? 'warning' : 'default'}
            size="large"
          >
            {result.confidenceLabel} Confidence
          </Badge>
        </div>

        <div className={styles.confidenceSection}>
          <div className={styles.confidenceHeader}>
            <span>Confidence Score</span>
            <span className={styles.confidenceValue}>
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
          <Progress
            value={confidence * 100}
            variant={isFake ? 'danger' : 'success'}
            size="large"
          />
        </div>

        <div className={styles.metaGrid}>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Processing Time</span>
            <span className={styles.metaValue}>{result.processingTime}</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Method</span>
            <span className={styles.metaValue}>{result.method || 'ensemble'}</span>
          </div>
          {result.riskLevel && (
            <div className={styles.metaItem}>
              <span className={styles.metaLabel}>Risk Level</span>
              <Badge variant={result.riskLevel === 'DANGER' ? 'danger' : 'success'}>
                {result.riskLevel}
              </Badge>
            </div>
          )}
        </div>

        {result.isEnsemble && result.ensembleDetails && (
          <div className={styles.ensembleSection}>
            <h3 className={styles.sectionTitle}>
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
              Ensemble Analysis
            </h3>
            <div className={styles.ensembleStats}>
              <div className={styles.statItem}>
                <span className={styles.statValue}>
                  {result.ensembleDetails.totalModels}
                </span>
                <span className={styles.statLabel}>Models</span>
              </div>
              <div className={styles.statItem}>
                <span className={styles.statValue}>
                  {result.ensembleDetails.realPercentage}%
                </span>
                <span className={styles.statLabel}>Real Vote</span>
              </div>
              <div className={styles.statItem}>
                <span className={styles.statValue}>
                  {result.ensembleDetails.agreementLevel}%
                </span>
                <span className={styles.statLabel}>Agreement</span>
              </div>
            </div>

            <div className={styles.modelsList}>
              {result.individualModels?.map((model, index) => (
                <div
                  key={model.modelName}
                  className={styles.modelItem}
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <div className={styles.modelInfo}>
                    <span className={styles.modelName}>{model.modelName}</span>
                    <Badge
                      variant={model.prediction === 'FAKE' ? 'fake' : 'real'}
                      size="small"
                    >
                      {model.prediction}
                    </Badge>
                  </div>
                  <div className={styles.modelStats}>
                    <div className={styles.modelConfidence}>
                      <span className={styles.confidenceBar}>
                        <span
                          className={styles.confidenceFill}
                          style={{ width: `${model.confidence * 100}%` }}
                        />
                      </span>
                      <span className={styles.confidenceText}>
                        {(model.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <span className={styles.contribution}>
                      {model.contributionPercentage?.toFixed(1)}% contribution
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className={styles.actions}>
          <Button variant="secondary" onClick={onReset}>
            Analyze Another
          </Button>
        </div>
      </Card>
    </div>
  );
}

export default AnalysisResult;