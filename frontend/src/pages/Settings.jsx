import { useEffect, useState } from "react";

import GlassCard from "../components/GlassCard.jsx";
import SectionHeader from "../components/SectionHeader.jsx";
import NeonButton from "../components/NeonButton.jsx";
import { fetchSettings, updateSettings } from "../services/api.js";

const Settings = () => {
  const [settings, setSettings] = useState({
    theme: "dark",
    notifications: true,
    animationMode: "high",
  });
  const [status, setStatus] = useState("");

  useEffect(() => {
    fetchSettings()
      .then((data) => setSettings(data.settings || settings))
      .catch(() => null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpdate = async () => {
    setStatus("");
    try {
      await updateSettings(settings);
      setStatus("Settings updated");
    } catch {
      setStatus("Update failed");
    }
  };

  return (
    <div className="space-y-8">
      <SectionHeader
        title="System Preferences"
        subtitle="Tune the visual intensity and notifications for your AI console."
      />
      <GlassCard className="space-y-4">
        <div className="space-y-2">
          <label className="text-xs uppercase tracking-[0.3em] text-textSecondary">
            Animation Mode
          </label>
          <select
            className="input-base w-full"
            value={settings.animationMode}
            onChange={(event) =>
              setSettings({ ...settings, animationMode: event.target.value })
            }
          >
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-textSecondary">
              Notifications
            </p>
            <p className="text-sm text-textSecondary">
              Toggle system alerts.
            </p>
          </div>
          <input
            type="checkbox"
            checked={settings.notifications}
            onChange={(event) =>
              setSettings({ ...settings, notifications: event.target.checked })
            }
          />
        </div>
        <NeonButton className="px-6 py-3" onClick={handleUpdate}>
          Save Changes
        </NeonButton>
        {status && <p className="text-sm text-textSecondary">{status}</p>}
      </GlassCard>
    </div>
  );
};

export default Settings;
