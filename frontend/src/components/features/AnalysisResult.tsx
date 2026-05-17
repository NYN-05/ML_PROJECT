import { useEffect, useState } from 'react'

interface ModelResult {
  modelName: string
  prediction: string
  confidence: number
  weight: number
  contributionPercentage: number
  inferenceTimeMs: number
}

interface EnsembleDetails {
  totalModels: number
  realVoteWeight: number
  fakeVoteWeight: number
  realPercentage: number
  agreementLevel: number
  totalInferenceTimeMs: number
}

interface Result {
  prediction: string
  confidence: number
  confidenceLabel: string
  riskLevel: string
  processingTime: string
  method: string
  isEnsemble: boolean
  ensembleDetails?: EnsembleDetails
  individualModels?: ModelResult[]
}

interface Props {
  result: Result
  onReset: () => void
}

export function AnalysisResult({ result, onReset }: Props) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    setTimeout(() => setVisible(true), 50)
  }, [])

  if (!result) return null

  const isFake = result.prediction === 'FAKE'
  const confidence = result.confidence || 0
  const confidencePercent = Math.round(confidence * 100)

  return (
    <div className={`glass-card rounded-2xl border border-border/50 overflow-hidden transition-all duration-500 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
      {/* Main Result Header */}
      <div className="p-6 md:p-8 border-b border-border/50">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            {/* Icon */}
            <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${
              isFake 
                ? 'bg-gradient-to-br from-rose-500/20 to-rose-600/10' 
                : 'bg-gradient-to-br from-emerald-500/20 to-emerald-600/10'
            }`}>
              {isFake ? (
                <svg className="w-7 h-7 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              ) : (
                <svg className="w-7 h-7 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              )}
            </div>

            {/* Prediction */}
            <div>
              <p className="text-sm text-text-muted">Prediction</p>
              <h2 className={`font-display text-2xl font-semibold ${
                isFake ? 'text-rose-400' : 'text-emerald-400'
              }`}>
                {isFake ? 'Likely Fake' : 'Likely Real'}
              </h2>
            </div>
          </div>

          {/* Confidence Badge */}
          <div className={`px-4 py-2 rounded-lg font-medium text-sm ${
            result.confidenceLabel === 'HIGH' 
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : result.confidenceLabel === 'MEDIUM'
              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
              : 'bg-text-muted/10 text-text-muted border border-text-muted/20'
          }`}>
            {result.confidenceLabel} Confidence
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mt-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-text-muted">Confidence Score</span>
            <span className="font-mono text-lg font-semibold text-text-primary">{confidencePercent}%</span>
          </div>
          <div className="h-2 bg-background-tertiary rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all duration-700 ${
                isFake 
                  ? 'bg-gradient-to-r from-rose-500 to-rose-400' 
                  : 'bg-gradient-to-r from-emerald-500 to-emerald-400'
              }`}
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Meta Info */}
      <div className="px-6 md:px-8 py-4 bg-background-secondary/50 flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-text-muted">Processing:</span>
          <span className="text-text-primary font-mono">{result.processingTime}</span>
        </div>
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          <span className="text-text-muted">Method:</span>
          <span className="text-text-primary font-mono">{result.method || 'ensemble'}</span>
        </div>
      </div>

      {/* Ensemble Details */}
      {result.isEnsemble && result.ensembleDetails && (
        <div className="px-6 md:px-8 py-6 border-t border-border/50">
          <h3 className="font-display font-medium text-text-primary mb-4 flex items-center gap-2">
            <svg className="w-5 h-5 text-accent-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            Ensemble Analysis
          </h3>
          
          {/* Stats Grid */}
          <div className="grid grid-cols-3 gap-3 mb-6">
            <div className="p-3 rounded-lg bg-background-tertiary text-center">
              <p className="text-2xl font-semibold text-accent-primary">{result.ensembleDetails.totalModels}</p>
              <p className="text-xs text-text-muted">Models</p>
            </div>
            <div className="p-3 rounded-lg bg-background-tertiary text-center">
              <p className="text-2xl font-semibold text-emerald-400">{result.ensembleDetails.realPercentage}%</p>
              <p className="text-xs text-text-muted">Real Vote</p>
            </div>
            <div className="p-3 rounded-lg bg-background-tertiary text-center">
              <p className="text-2xl font-semibold text-accent-cyan">{result.ensembleDetails.agreementLevel}%</p>
              <p className="text-xs text-text-muted">Agreement</p>
            </div>
          </div>

          {/* Individual Model Results */}
          {result.individualModels && (
            <div className="space-y-2">
              <p className="text-xs text-text-muted uppercase tracking-wide mb-3">Individual Model Predictions</p>
              {result.individualModels.map((model, index) => (
                <div 
                  key={model.modelName}
                  className="flex items-center justify-between p-3 rounded-lg bg-background-tertiary/50 hover:bg-background-tertiary transition-colors"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <div className="flex items-center gap-3">
                    <span className={`w-2 h-2 rounded-full ${model.prediction === 'FAKE' ? 'bg-rose-500' : 'bg-emerald-500'}`} />
                    <span className="text-sm font-medium text-text-primary">{model.modelName}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-1.5 bg-background-hover rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full ${model.prediction === 'FAKE' ? 'bg-rose-500' : 'bg-emerald-500'}`}
                          style={{ width: `${model.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-text-muted font-mono w-10">
                        {Math.round(model.confidence * 100)}%
                      </span>
                    </div>
                    <span className="text-xs text-text-muted w-16 text-right">
                      {model.contributionPercentage?.toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Action */}
      <div className="px-6 md:px-8 py-6 border-t border-border/50 bg-background-secondary/30">
        <button 
          onClick={onReset}
          className="btn-secondary w-full"
        >
          Analyze Another
        </button>
      </div>
    </div>
  )
}

export default AnalysisResult