import { sanitizeText } from "./sanitize.js";

const MAX_TEXT_LENGTH = 20000;

export const validateAnalyzeRequest = (payload) => {
  if (!payload || typeof payload !== "object") {
    return { ok: false, message: "Invalid request body" };
  }

  const title = typeof payload.title === "string" ? payload.title.trim() : "";
  const text = typeof payload.text === "string" ? payload.text.trim() : "";

  if (!title || !text) {
    return { ok: false, message: "Input text cannot be empty" };
  }

  if (text.length > MAX_TEXT_LENGTH) {
    return { ok: false, message: "Input text exceeds size limit" };
  }

  return {
    ok: true,
    data: {
      title: sanitizeText(title),
      text: sanitizeText(text),
    },
  };
};

export const validateSettingsPatch = (payload) => {
  if (!payload || typeof payload !== "object") {
    return { ok: false, message: "Invalid request body" };
  }

  const patch = {};
  if (typeof payload.theme === "string") {
    patch.theme = payload.theme;
  }
  if (typeof payload.notifications === "boolean") {
    patch.notifications = payload.notifications;
  }
  if (typeof payload.animationMode === "string") {
    patch.animationMode = payload.animationMode;
  }

  if (Object.keys(patch).length === 0) {
    return { ok: false, message: "No valid settings provided" };
  }

  return { ok: true, data: patch };
};
