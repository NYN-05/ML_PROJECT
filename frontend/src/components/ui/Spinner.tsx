import styles from './Spinner.module.css';

export function Spinner({ size = 'medium', className = '' }) {
  const spinnerClasses = [styles.spinner, styles[size], className]
    .filter(Boolean)
    .join(' ');

  return <div className={spinnerClasses} />;
}

export default Spinner;