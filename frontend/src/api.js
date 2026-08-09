const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8001";
const DEFAULT_TIMEOUT_MS = 15000;

function withTimeout(ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { controller, timer };
}

async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail || payload?.message || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

export async function apiRequest(path, { timeoutMs = DEFAULT_TIMEOUT_MS, ...options } = {}) {
  const { controller, timer } = withTimeout(timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...options, signal: controller.signal });
    return await parseResponse(response);
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Request timed out. Please try again.");
    }
    if (error instanceof TypeError) {
      throw new Error("Network error while contacting the API.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  dashboard: () => apiRequest("/dashboard"),
  replay: () => apiRequest("/replay"),
  replayOptions: () => apiRequest("/replay/options"),
  diagnostics: () => apiRequest("/diagnostics"),
  mlStatus: () => apiRequest("/ml/status"),
  analytics: (task) => apiRequest(`/analytics?task=${task}`),
  startReplay: (payload) =>
    apiRequest("/replay/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  controlReplay: (action) =>
    apiRequest("/replay/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    }),
  previewUpload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return apiRequest("/uploads/preview", { method: "POST", body: form, timeoutMs: 45000 });
  },
};
