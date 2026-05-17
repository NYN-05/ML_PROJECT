import { forwardRef } from 'react';
import styles from './Input.module.css';

export const Input = forwardRef(function Input(
  {
    label,
    error,
    icon,
    fullWidth = true,
    className = '',
    ...props
  },
  ref
) {
  const inputClasses = [
    styles.input,
    icon && styles.hasIcon,
    error && styles.hasError,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={`${styles.wrapper} ${fullWidth ? styles.fullWidth : ''}`}>
      {label && <label className={styles.label}>{label}</label>}
      <div className={styles.inputWrapper}>
        {icon && <span className={styles.icon}>{icon}</span>}
        <input ref={ref} className={inputClasses} {...props} />
      </div>
      {error && <span className={styles.error}>{error}</span>}
    </div>
  );
});

export const Textarea = forwardRef(function Textarea(
  { label, error, fullWidth = true, className = '', rows = 5, ...props },
  ref
) {
  const textareaClasses = [
    styles.textarea,
    error && styles.hasError,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={`${styles.wrapper} ${fullWidth ? styles.fullWidth : ''}`}>
      {label && <label className={styles.label}>{label}</label>}
      <textarea
        ref={ref}
        className={textareaClasses}
        rows={rows}
        {...props}
      />
      {error && <span className={styles.error}>{error}</span>}
    </div>
  );
});

export default Input;