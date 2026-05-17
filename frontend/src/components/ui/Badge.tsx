import styles from './Badge.module.css';

export function Badge({
  children,
  variant = 'default',
  size = 'medium',
  dot = false,
  className = '',
  ...props
}) {
  const badgeClasses = [
    styles.badge,
    styles[variant],
    styles[size],
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <span className={badgeClasses} {...props}>
      {dot && <span className={styles.dot} />}
      {children}
    </span>
  );
}

export default Badge;