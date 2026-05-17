import { sanitizeText } from "../utils/sanitize.js";
import { makeAnalysisId, toConfidenceLabel, toRiskLevel } from "../utils/response.js";

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL || "http://localhost:8001/predict";

const TIMEOUT_MS = 2000;

const parsePrediction = (payload) => {
  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid ML response");
  }

  const prediction = payload.prediction;
  const confidence = Number(payload.confidence);
  if (!prediction || Number.isNaN(confidence)) {
    throw new Error("Invalid ML response");
  }

  return { prediction, confidence };
};

const callMlService = async (title, text) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(ML_SERVICE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ title, text }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const error = new Error("AI service offline");
      error.code = "MODEL_UNAVAILABLE";
      throw error;
    }

    return parsePrediction(await response.json());
  } finally {
    clearTimeout(timeout);
  }
};

export const runAnalysis = async ({ title, text }) => {
  const sanitizedTitle = sanitizeText(title);
  const sanitizedText = sanitizeText(text);

  const start = performance.now();
  const { prediction, confidence } = await callMlService(
    sanitizedTitle,
    sanitizedText
  );
  const duration = Math.max(0, performance.now() - start);

  const payload = {
    success: true,
    prediction,
    confidence,
    confidenceLabel: toConfidenceLabel(confidence),
    riskLevel: toRiskLevel(prediction),
    processingTime: `${Math.round(duration)}ms`,
    timestamp: new Date().toISOString(),
    analysisId: makeAnalysisId(),
  };

  const historyEntry = {
    analysisId: payload.analysisId,
    prediction: payload.prediction,
    confidence: payload.confidence,
    timestamp: payload.timestamp,
    title: sanitizedTitle.slice(0, 120),
  };

  return { payload, historyEntry };
};
