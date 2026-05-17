import { motion } from "framer-motion";

const NeonButton = ({ children, className = "", variant = "primary", ...props }) => {
  const baseClass =
    variant === "primary" ? "button-primary" : "button-secondary";

  return (
    <motion.button
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.98 }}
      className={`${baseClass} ${className}`}
      {...props}
    >
      {children}
    </motion.button>
  );
};

export default NeonButton;
