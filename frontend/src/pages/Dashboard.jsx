import { useEffect, useState } from "react";

import GlassCard from "../components/GlassCard.jsx";
import SectionHeader from "../components/SectionHeader.jsx";
import Topbar from "../components/Topbar.jsx";
import { fetchHistory } from "../services/api.js";

const Dashboard = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchHistory()
      .then((data) => setHistory(data.history || []))
      .catch(() => setHistory([]));
  }, []);

  const stats = [
    { label: "Total Analyses", value: history.length },
    { label: "Latest Confidence", value: history[0]?.confidence || 0 },
    { label: "System Status", value: "Online" },
  ];

  return (
    <div className="space-y-8">
      <Topbar />
      <SectionHeader
        title="Threat Intelligence Overview"
        subtitle="Your AI command center summarizes recent analyses, confidence signals, and system readiness."
      />
      <div className="grid gap-6 md:grid-cols-3">
        {stats.map((stat) => (
          <GlassCard key={stat.label} className="space-y-2">
            <p className="text-xs uppercase tracking-[0.3em] text-textSecondary">
              {stat.label}
            </p>
            <p className="text-3xl font-orbitron text-textPrimary">
              {stat.value}
            </p>
          </GlassCard>
        ))}
      </div>
      <GlassCard className="space-y-3">
        <p className="text-xs uppercase tracking-[0.3em] text-textSecondary">
          Latest Activity
        </p>
        {history.length === 0 ? (
          <p className="text-sm text-textSecondary">
            No history yet. Run your first scan in the Analysis module.
          </p>
        ) : (
          <div className="space-y-3">
            {history.slice(0, 3).map((item) => (
              <div
                key={item.analysisId}
                className="flex items-center justify-between border-b border-white/10 pb-3"
              >
                <div>
                  <p className="text-sm text-textPrimary">
                    {item.title || "Untitled signal"}
                  </p>
                  <p className="text-xs text-textSecondary">{item.timestamp}</p>
                </div>
                <div className="text-sm font-jetbrains text-neon">
                  {item.prediction} {Math.round(item.confidence * 100)}%
                </div>
              </div>
            ))}
          </div>
        )}
      </GlassCard>
    </div>
  );
};

export default Dashboard;
