import { useEffect, useState } from "react";

import GlassCard from "../components/GlassCard.jsx";
import SectionHeader from "../components/SectionHeader.jsx";
import { fetchHistory } from "../services/api.js";

const History = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchHistory()
      .then((data) => setHistory(data.history || []))
      .catch(() => setHistory([]));
  }, []);

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Analysis History"
        subtitle="Every scan is logged with prediction confidence and timestamp for rapid review."
      />
      <GlassCard className="space-y-4">
        <div className="grid grid-cols-4 text-xs uppercase tracking-[0.3em] text-textSecondary">
          <span>Prediction</span>
          <span>Confidence</span>
          <span>Timestamp</span>
          <span>Title</span>
        </div>
        {history.length === 0 ? (
          <p className="text-sm text-textSecondary">No history yet.</p>
        ) : (
          <div className="space-y-3">
            {history.map((item) => (
              <div
                key={item.analysisId}
                className="grid grid-cols-4 gap-4 border-b border-white/10 pb-3 text-sm"
              >
                <span className="font-jetbrains text-neon">
                  {item.prediction}
                </span>
                <span className="font-jetbrains text-textPrimary">
                  {Math.round(item.confidence * 100)}%
                </span>
                <span className="text-textSecondary">{item.timestamp}</span>
                <span className="text-textSecondary">
                  {item.title || "Untitled signal"}
                </span>
              </div>
            ))}
          </div>
        )}
      </GlassCard>
    </div>
  );
};

export default History;
