const SCRIPT_TAG_RE = /<\/?script[^>]*>/gi;
const HTML_TAG_RE = /<[^>]+>/g;

export const sanitizeText = (value) => {
  if (typeof value !== "string") {
    return "";
  }

  return value
    .replace(SCRIPT_TAG_RE, " ")
    .replace(HTML_TAG_RE, " ")
    .replace(/\s+/g, " ")
    .trim();
};
