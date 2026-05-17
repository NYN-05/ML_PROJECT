import { useState } from "react";

import GlassCard from "../components/GlassCard.jsx";
import SectionHeader from "../components/SectionHeader.jsx";
import ConfidenceMeter from "../components/ConfidenceMeter.jsx";
import AIProcessingLoader from "../components/AIProcessingLoader.jsx";
import NeonButton from "../components/NeonButton.jsx";
import { analyzeNews } from "../services/api.js";

const Analysis = () => {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const data = await analyzeNews({ title, text });
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.error?.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Live Analysis"
        subtitle="Feed the neural core with a headline and article text. The system will return trust signals and calibrated confidence."
      />
      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <GlassCard className="space-y-4">
          <form className="space-y-4" onSubmit={handleSubmit}>
            <input
              className="input-base w-full"
              placeholder="Headline"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
            />
            <textarea
              className="input-base w-full min-h-[200px]"
              placeholder="Paste the full article text here..."
              value={text}
              onChange={(event) => setText(event.target.value)}
            />
            <NeonButton className="w-full py-3" type="submit">
              Analyze Signal
            </NeonButton>
          </form>
          {error && <p className="text-sm text-warning">{error}</p>}
        </GlassCard>
        <div className="space-y-4">
          {loading && <AIProcessingLoader />}
          {result && (
            <GlassCard className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-textSecondary">
                    Prediction
                  </p>
                  <p className="text-2xl font-orbitron text-textPrimary">
                    {result.prediction}
                  </p>
                </div>
                <div
                  className={`rounded-full px-4 py-1 text-xs font-jetbrains ${
                    result.prediction === "REAL"
                      ? "bg-success/20 text-success"
                      : result.prediction === "FAKE"
                      ? "bg-warning/20 text-warning"
                      : "bg-amber/20 text-amber"
                  }`}
                >
                  {result.riskLevel} risk
                </div>
              </div>
              <ConfidenceMeter
                value={result.confidence}
                label={result.confidenceLabel}
              />
              <div className="text-xs text-textSecondary">
                Processing time: {result.processingTime}
              </div>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;
