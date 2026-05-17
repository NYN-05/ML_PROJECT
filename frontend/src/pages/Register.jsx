import { Link } from "react-router-dom";
import { motion } from "framer-motion";

import BackgroundShell from "../components/BackgroundShell.jsx";
import GlassCard from "../components/GlassCard.jsx";
import NeonButton from "../components/NeonButton.jsx";

const Register = () => (
  <BackgroundShell>
    <div className="min-h-screen flex items-center justify-center px-6 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <GlassCard className="space-y-6">
          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.3em] text-neon">Onboard</p>
            <h2 className="text-2xl font-orbitron text-textPrimary">Create Profile</h2>
            <p className="text-sm text-textSecondary">
              Authentication is disabled for this project build.
            </p>
          </div>
          <form className="space-y-4">
            <input className="input-base w-full" placeholder="Username" />
            <input className="input-base w-full" placeholder="Email" />
            <input
              className="input-base w-full"
              placeholder="Password"
              type="password"
            />
            <NeonButton className="w-full py-3">Register</NeonButton>
          </form>
          <p className="text-sm text-textSecondary">
            Already in system?{" "}
            <Link className="text-neon" to="/login">
              Login
            </Link>
          </p>
        </GlassCard>
      </motion.div>
    </div>
  </BackgroundShell>
);

export default Register;
