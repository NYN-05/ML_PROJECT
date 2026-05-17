import { Router } from "express";

import { loadSettings, saveSettings } from "../store/settingsStore.js";
import { validateSettingsPatch } from "../utils/validation.js";
import { errorResponse } from "../utils/response.js";

const router = Router();

router.get("/settings", async (_req, res) => {
  const settings = await loadSettings();
  res.json({ success: true, settings });
});

router.patch("/settings", async (req, res) => {
  const validation = validateSettingsPatch(req.body);
  if (!validation.ok) {
    return res.status(400).json(
      errorResponse("INVALID_INPUT", validation.message)
    );
  }

  await saveSettings(validation.data);
  res.json({ success: true, message: "Settings updated successfully" });
});

export default router;
