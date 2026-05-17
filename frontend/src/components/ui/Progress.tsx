import styles from './Progress.module.css';

export function Progress({
  value = 0,
  max = 100,
  variant = 'default',
  size = 'medium',
  showLabel = false,
  animated = true,
  className = '',
}) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const progressClasses = [
    styles.progress,
    styles[variant],
    styles[size],
    animated && styles.animated,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={styles.wrapper}>
      <div className={progressClasses}>
        <div
          className={styles.bar}
          style={{ width: `${percentage}%` }}
          role="progressbar"
          aria-valuenow={value}
          aria-valuemin={0}
          aria-valuemax={max}
        />
      </div>
      {showLabel && (
        <span className={styles.label}>{Math.round(percentage)}%</span>
      )}
    </div>
  );
}

export default Progress;