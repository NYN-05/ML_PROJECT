const GlassCard = ({ children, className = "" }) => (
  <div className={`glass-panel rounded-[20px] p-6 ${className}`}>{children}</div>
);

export default GlassCard;
