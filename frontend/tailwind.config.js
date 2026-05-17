/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: "#0B1020",
        surface: "#111827",
        elevated: "#1A2336",
        neon: "#00F5FF",
        electric: "#8B5CF6",
        warning: "#FF4D6D",
        success: "#22C55E",
        amber: "#F59E0B",
        textPrimary: "#F3F4F6",
        textSecondary: "#9CA3AF",
        textMuted: "#6B7280",
      },
      fontFamily: {
        orbitron: ["Orbitron", "sans-serif"],
        inter: ["Inter", "sans-serif"],
        jetbrains: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        glowCyan: "0 0 24px rgba(0, 245, 255, 0.25)",
        glowPurple: "0 0 24px rgba(139, 92, 246, 0.25)",
        glowRed: "0 0 22px rgba(255, 77, 109, 0.35)",
        glowGreen: "0 0 22px rgba(34, 197, 94, 0.3)",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { opacity: 0.35 },
          "50%": { opacity: 1 },
        },
        sweep: {
          "0%": { transform: "translateX(-60%)" },
          "100%": { transform: "translateX(160%)" },
        },
      },
      animation: {
        pulseGlow: "pulseGlow 1.6s ease-in-out infinite",
        sweep: "sweep 2s linear infinite",
      },
    },
  },
  plugins: [],
};
