import styles from './Skeleton.module.css';

export function Skeleton({
  width,
  height,
  variant = 'text',
  className = '',
}) {
  const skeletonClasses = [styles.skeleton, styles[variant], className]
    .filter(Boolean)
    .join(' ');

  const style = {
    width: width || (variant === 'text' ? '100%' : undefined),
    height: height || (variant === 'text' ? '1em' : undefined),
  };

  return <div className={skeletonClasses} style={style} />;
}

export default Skeleton;