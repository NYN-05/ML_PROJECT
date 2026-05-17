import { useState, useEffect } from 'react'
import { useAnalysis } from '../hooks/useAnalysis'
import { AnalysisForm } from '../components/features/AnalysisForm'
import { AnalysisResult } from '../components/features/AnalysisResult'

export function HomePage() {
  const { isLoading, error, result, analyze, clearResult } = useAnalysis()
  const [showResult, setShowResult] = useState(false)

  useEffect(() => {
    if (result) {
      setShowResult(true)
    }
  }, [result])

  const handleSubmit = async (text: string, title: string) => {
    try {
      await analyze(text, title)
    } catch (err) {
      console.error('Analysis failed:', err)
    }
  }

  const handleReset = () => {
    clearResult()
    setShowResult(false)
  }

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background-primary/80 backdrop-blur-xl">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary to-accent-cyan flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
            </div>
            <span className="font-display font-semibold text-lg text-text-primary">FakeScope</span>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-xs font-medium text-emerald-400">API Connected</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-24 pb-16 px-6">
        <div className="max-w-3xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12 animate-fade-in">
            <h1 className="font-display text-4xl md:text-5xl font-semibold text-text-primary mb-4 tracking-tight">
              Detect <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-primary to-accent-cyan">Misinformation</span> with AI
            </h1>
            <p className="text-text-secondary text-lg max-w-xl mx-auto">
              Enterprise-grade fake news detection powered by FakeBERT ensemble models. 
              Analyze news articles with precision and confidence.
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center gap-3 animate-slide-up">
              <svg className="w-5 h-5 text-rose-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="text-rose-300 text-sm">{error}</span>
            </div>
          )}

          {/* Form or Result */}
          {!showResult ? (
            <div className="animate-slide-up" style={{ animationDelay: '0.1s' }}>
              <AnalysisForm onSubmit={handleSubmit} isLoading={isLoading} />
            </div>
          ) : (
            <div className="animate-slide-up">
              <AnalysisResult result={result} onReset={handleReset} />
            </div>
          )}

          {/* Features Grid */}
          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-4 animate-fade-in" style={{ animationDelay: '0.2s' }}>
            {[
              { icon: '🧠', title: 'FakeBERT Models', desc: '7 BERT-based models for accurate detection' },
              { icon: '⚡', title: 'Real-time Analysis', desc: 'Get results in seconds, not minutes' },
              { icon: '📊', title: 'Detailed Insights', desc: 'View individual model predictions' },
            ].map((feature, i) => (
              <div 
                key={i}
                className="p-5 rounded-xl bg-background-card border border-border/50 hover:border-border-light transition-all duration-300 group"
              >
                <span className="text-2xl mb-3 block">{feature.icon}</span>
                <h3 className="font-display font-medium text-text-primary mb-1">{feature.title}</h3>
                <p className="text-sm text-text-muted">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/30 py-6 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-text-muted">
          <span>© 2026 FakeScope. AI-Powered News Verification.</span>
          <span className="font-mono text-xs">Powered by FakeBERT Ensemble</span>
        </div>
      </footer>
    </div>
  )
}

export default HomePage