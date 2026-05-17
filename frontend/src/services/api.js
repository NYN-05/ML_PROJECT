import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:5000/api/v1",
  timeout: 5000,
});

export const analyzeNews = async (payload) => {
  const { data } = await api.post("/analyze", payload);
  return data;
};

export const fetchHistory = async () => {
  const { data } = await api.get("/history");
  return data;
};

export const fetchSettings = async () => {
  const { data } = await api.get("/settings");
  return data;
};

export const updateSettings = async (payload) => {
  const { data } = await api.patch("/settings", payload);
  return data;
};
