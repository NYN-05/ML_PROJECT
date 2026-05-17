import express from "express";
import cors from "cors";
import dotenv from "dotenv";

import analyzeRouter from "./routes/analyze.js";
import historyRouter from "./routes/history.js";
import settingsRouter from "./routes/settings.js";
import { errorResponse } from "./utils/response.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: "1mb" }));

app.get("/api/v1/health", (_req, res) => {
  res.json({ success: true, status: "ok" });
});

app.use("/api/v1", analyzeRouter);
app.use("/api/v1", historyRouter);
app.use("/api/v1", settingsRouter);

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json(
    errorResponse("SERVER_ERROR", "Internal backend failure")
  );
});

app.listen(port, () => {
  console.log(`Backend running on port ${port}`);
});
