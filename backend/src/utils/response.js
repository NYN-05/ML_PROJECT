export const errorResponse = (code, message) => ({
  success: false,
  error: {
    code,
    message,
  },
});

export const toConfidenceLabel = (confidence) => {
  if (confidence >= 0.9) {
    return "High Confidence";
  }
  if (confidence >= 0.7) {
    return "Medium Confidence";
  }
  return "Low Confidence";
};

export const toRiskLevel = (prediction) => {
  if (prediction === "FAKE") {
    return "High";
  }
  if (prediction === "REAL") {
    return "Low";
  }
  return "Unknown";
};

export const makeAnalysisId = () => {
  const suffix = Math.floor(Math.random() * 100000)
    .toString()
    .padStart(5, "0");
  return `ANL_${Date.now()}_${suffix}`;
};
