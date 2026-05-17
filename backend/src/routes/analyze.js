import { Router } from "express";

import { runAnalysis } from "../services/analysisService.js";
import { validateAnalyzeRequest } from "../utils/validation.js";
import { errorResponse } from "../utils/response.js";
import { saveHistoryEntry } from "../store/historyStore.js";

const router = Router();

router.post("/analyze", async (req, res) => {
  const validation = validateAnalyzeRequest(req.body);
  if (!validation.ok) {
    return res.status(400).json(
      errorResponse("INVALID_INPUT", validation.message)
    );
  }

  try {
    const result = await runAnalysis(validation.data);
    await saveHistoryEntry(result.historyEntry);
    return res.json(result.payload);
  } catch (error) {
    const code = error.code || "MODEL_UNAVAILABLE";
    const message = error.message || "AI service offline";
    return res.status(503).json(errorResponse(code, message));
  }
});

export default router;
