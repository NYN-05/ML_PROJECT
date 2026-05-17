import { Router } from "express";

import { loadHistory } from "../store/historyStore.js";

const router = Router();

router.get("/history", async (_req, res) => {
  const history = await loadHistory();
  res.json({ success: true, history });
});

export default router;
