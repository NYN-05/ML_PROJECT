import { useState } from 'react';
import { Textarea, Button, Card } from '../ui';
import styles from './AnalysisForm.module.css';

export function AnalysisForm({ onSubmit, isLoading }) {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    if (text.length < 10) {
      setError('Text must be at least 10 characters');
      return;
    }

    await onSubmit(text, title);
  };

  return (
    <Card className={styles.formCard} padding="large">
      <form onSubmit={handleSubmit}>
        <div className={styles.header}>
          <div className={styles.iconWrapper}>
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
          </div>
          <div>
            <h2 className={styles.title}>Analyze Content</h2>
            <p className={styles.subtitle}>
              Enter news text to detect potential misinformation
            </p>
          </div>
        </div>

        <div className={styles.fields}>
          <Textarea
            label="Article Title (optional)"
            placeholder="Enter the headline or title of the article..."
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            rows={2}
          />

          <div className={styles.textareaWrapper}>
            <Textarea
              label="Article Content"
              placeholder="Paste the full article text or news content you want to analyze for authenticity..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={8}
              error={error}
            />
            <span className={styles.charCount}>
              {text.length} characters
            </span>
          </div>
        </div>

        <div className={styles.actions}>
          <Button
            type="submit"
            variant="primary"
            size="large"
            fullWidth
            isLoading={isLoading}
            disabled={isLoading}
          >
            {isLoading ? 'Analyzing...' : 'Analyze for Fake News'}
          </Button>
        </div>

        <p className={styles.disclaimer}>
          Powered by an ensemble of 7 AI models with weighted voting
        </p>
      </form>
    </Card>
  );
}

export default AnalysisForm;