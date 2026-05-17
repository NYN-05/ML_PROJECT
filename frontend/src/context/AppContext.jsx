import { createContext, useContext, useMemo, useState } from "react";

const AppContext = createContext(null);

const DEFAULT_SETTINGS = {
  theme: "dark",
  notifications: true,
  animationMode: "high",
};

export const AppProvider = ({ children }) => {
  const [user, setUser] = useState({ name: "Analyst" });
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [history, setHistory] = useState([]);

  const value = useMemo(
    () => ({
      user,
      setUser,
      settings,
      setSettings,
      history,
      setHistory,
    }),
    [user, settings, history]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("AppContext is not available");
  }
  return context;
};
