const BackgroundShell = ({ children }) => (
  <div className="relative min-h-screen overflow-hidden bg-base text-textPrimary">
    <div className="absolute inset-0 grid-overlay" />
    <div className="absolute -top-24 right-0 h-72 w-72 rounded-full bg-electric/20 blur-[120px]" />
    <div className="absolute bottom-0 left-0 h-72 w-72 rounded-full bg-neon/20 blur-[120px]" />
    <div className="relative z-10">{children}</div>
  </div>
);

export default BackgroundShell;
