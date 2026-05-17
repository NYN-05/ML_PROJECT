import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const STORE_PATH =
  process.env.SETTINGS_STORE_PATH ||
  path.join(__dirname, "../data/settings.json");

const DEFAULT_SETTINGS = {
  theme: "dark",
  notifications: true,
  animationMode: "high",
};

const ensureStore = async () => {
  try {
    await fs.access(STORE_PATH);
  } catch {
    await fs.mkdir(path.dirname(STORE_PATH), { recursive: true });
    await fs.writeFile(
      STORE_PATH,
      JSON.stringify(DEFAULT_SETTINGS, null, 2),
      "utf-8"
    );
  }
};

export const loadSettings = async () => {
  await ensureStore();
  const raw = await fs.readFile(STORE_PATH, "utf-8");
  return JSON.parse(raw);
};

export const saveSettings = async (patch) => {
  await ensureStore();
  const current = await loadSettings();
  const next = { ...current, ...patch };
  await fs.writeFile(STORE_PATH, JSON.stringify(next, null, 2));
};
