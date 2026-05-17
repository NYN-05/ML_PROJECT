import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ShieldCheck, ScanEye, Sparkles } from "lucide-react";

import BackgroundShell from "../components/BackgroundShell.jsx";
import NeonButton from "../components/NeonButton.jsx";
import GlassCard from "../components/GlassCard.jsx";

const features = [
  {
    icon: ShieldCheck,
    title: "Trust Signal",
    description: "Classify information with calibrated confidence signals.",
  },
  {
    icon: ScanEye,
    title: "Neural Scan",
    description: "Realtime LSTM analysis for high-signal decisions.",
  },
  {
    icon: Sparkles,
    title: "Immersive UI",
    description: "Cyberpunk interface tuned for focus and clarity.",
  },
];

const Landing = () => (
  <BackgroundShell>
    <section className="min-h-screen px-6 py-16">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="space-y-6"
        >
          <p className="text-xs uppercase tracking-[0.4em] text-neon">
            Fake News Detection System
          </p>
          <h1 className="text-4xl md:text-6xl font-orbitron text-textPrimary leading-tight">
            Futuristic Intelligence for Verifying Reality.
          </h1>
          <p className="text-lg text-textSecondary max-w-2xl">
            Submit any article and the neural core will evaluate authenticity,
            risk signals, and calibrated confidence in seconds.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link to="/analysis">
              <NeonButton className="px-6 py-3">Start Analysis</NeonButton>
            </Link>
            <Link to="/dashboard">
              <NeonButton variant="secondary" className="px-6 py-3">
                View Dashboard
              </NeonButton>
            </Link>
          </div>
        </motion.div>
        <div className="grid gap-6 md:grid-cols-3">
          {features.map(({ icon: Icon, title, description }) => (
            <GlassCard key={title} className="space-y-3">
              <Icon className="text-neon" />
              <h3 className="text-lg font-orbitron text-textPrimary">{title}</h3>
              <p className="text-sm text-textSecondary">{description}</p>
            </GlassCard>
          ))}
        </div>
      </div>
    </section>
  </BackgroundShell>
);

export default Landing;
