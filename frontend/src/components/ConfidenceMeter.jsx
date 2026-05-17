import { motion } from "framer-motion";

const ConfidenceMeter = ({ value = 0, label = "" }) => {
  const percentage = Math.min(100, Math.max(0, Math.round(value * 100)));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-textSecondary">
        <span className="font-jetbrains tracking-wide">Confidence</span>
        <span className="font-jetbrains">{percentage}%</span>
      </div>
      <div className="h-2 rounded-full bg-white/10 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.6 }}
          className="h-full bg-gradient-to-r from-neon via-cyan-300 to-electric"
        />
      </div>
      {label && (
        <div className="text-xs text-textSecondary font-jetbrains">{label}</div>
      )}
    </div>
  );
};

export default ConfidenceMeter;
