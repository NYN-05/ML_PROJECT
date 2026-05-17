import { motion } from "framer-motion";

const AIProcessingLoader = () => (
  <div className="glass-panel rounded-[20px] p-6 relative overflow-hidden">
    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-neon/10 to-transparent animate-sweep" />
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-textSecondary">AI Processing</p>
        <p className="text-lg font-orbitron text-textPrimary">Neural Scan Active</p>
      </div>
      <motion.div
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{ duration: 1.4, repeat: Infinity }}
        className="h-3 w-3 rounded-full bg-neon shadow-glowCyan"
      />
    </div>
  </div>
);

export default AIProcessingLoader;
