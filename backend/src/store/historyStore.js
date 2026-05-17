import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const STORE_PATH =
  process.env.HISTORY_STORE_PATH || path.join(__dirname, "../data/history.json");

const ensureStore = async () => {
  try {
    await fs.access(STORE_PATH);
  } catch {
    await fs.mkdir(path.dirname(STORE_PATH), { recursive: true });
    await fs.writeFile(STORE_PATH, JSON.stringify([]), "utf-8");
  }
};

export const loadHistory = async () => {
  await ensureStore();
  const raw = await fs.readFile(STORE_PATH, "utf-8");
  return JSON.parse(raw);
};

export const saveHistoryEntry = async (entry) => {
  await ensureStore();
  const history = await loadHistory();
  history.unshift(entry);
  await fs.writeFile(STORE_PATH, JSON.stringify(history.slice(0, 200), null, 2));
};
