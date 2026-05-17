const SectionHeader = ({ title, subtitle }) => (
  <div className="space-y-2">
    <h2 className="text-2xl font-orbitron text-textPrimary">{title}</h2>
    {subtitle && (
      <p className="text-sm text-textSecondary leading-relaxed max-w-2xl">
        {subtitle}
      </p>
    )}
  </div>
);

export default SectionHeader;
