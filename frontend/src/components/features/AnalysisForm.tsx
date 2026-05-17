import { useState, FormEvent } from 'react'

interface Props {
  onSubmit: (text: string, title: string) => Promise<void>
  isLoading: boolean
}

export function AnalysisForm({ onSubmit, isLoading }: Props) {
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')

    if (!text.trim()) {
      setError('Please enter some text to analyze')
      return
    }

    if (text.length < 10) {
      setError('Text must be at least 10 characters')
      return
    }

    if (text.length > 20000) {
      setError('Text must be less than 20,000 characters')
      return
    }

    await onSubmit(text, title)
  }

  return (
    <div className="glass-card rounded-2xl p-6 md:p-8 border border-border/50">
      <form onSubmit={handleSubmit}>
        {/* Header */}
        <div className="flex items-start gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-primary to-accent-cyan flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h2 className="font-display font-semibold text-xl text-text-primary">Analyze Content</h2>
            <p className="text-sm text-text-muted">Enter news text to detect potential misinformation</p>
          </div>
        </div>

        {/* Fields */}
        <div className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Article Title <span className="text-text-muted">(optional)</span>
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter the headline or title..."
              className="input-field"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Article Content <span className="text-rose-400">*</span>
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste the full article text you want to analyze for authenticity..."
              rows={7}
              className="input-field resize-none"
            />
            <div className="flex justify-between mt-2">
              <span className={`text-xs ${error ? 'text-rose-400' : 'text-text-muted'}`}>
                {error || `${text.length} characters`}
              </span>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading}
          className="w-full mt-6 btn-primary flex items-center justify-center gap-2 h-12 text-base"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Analyzing...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Analyze for Fake News
            </>
          )}
        </button>

        {/* Footer Note */}
        <p className="text-center text-xs text-text-muted mt-4">
          Powered by FakeBERT ensemble of 7 AI models with weighted voting
        </p>
      </form>
    </div>
  )
}

export default AnalysisForm