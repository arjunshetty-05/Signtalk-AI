// Axios helpers for the SignTalk v3 HTTP API (Section 6).
//
// Every call targets the same-origin `/api/*` prefix; in dev Vite proxies that
// to the FastAPI server (see vite.config.js), so there is no backend host
// hard-coded here and no CORS handling needed.

import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

/**
 * POST /api/recognize — recognise a whole clip (Section 6.1, JSON branch).
 *
 * @param {{frames: string[], fps: number, signer_id: string,
 *          scenario_id?: string|null}} payload
 *   `frames` are base64 JPEG strings of the UN-mirrored capture, one per entry.
 * @returns {Promise<object>} the Section 6.1 decision dict.
 */
export async function recognizeClip(payload) {
  const { data } = await api.post("/recognize", payload);
  return data;
}

/**
 * POST /api/confirm — record the user's tap on a candidate chip (Section 6.2).
 *
 * @param {{clip_id: string, chosen_label: string}} payload
 * @returns {Promise<{clip_id: string, chosen_label: string, stored: boolean}>}
 */
export async function confirmChoice(payload) {
  const { data } = await api.post("/confirm", payload);
  return data;
}

/**
 * POST /api/compose — build a sentence (en/hi/kn) from signed words (Section 6.3).
 *
 * @param {{words: string[], emotion?: string, history?: string[],
 *          scenario_id?: string|null}} payload
 * @returns {Promise<{sentences: {en: string, hi: string, kn: string},
 *          source: string, verified: boolean, latency_ms: number}>}
 */
export async function composeSentence(payload) {
  const { data } = await api.post("/compose", payload);
  return data;
}

/**
 * GET /api/vocab — the demo vocabulary (Section 6.6).
 * @returns {Promise<Array<{label: string, category: string, demo: boolean}>>}
 */
export async function getVocab() {
  const { data } = await api.get("/vocab");
  return data;
}

/**
 * POST /api/enroll/start — begin an enrollment session for a signer.
 * @param {string} signerId
 * @returns {Promise<{signer_id: string, session_id: string}>}
 */
export async function enrollStart(signerId) {
  const { data } = await api.post("/enroll/start", { signer_id: signerId });
  return data;
}

/**
 * POST /api/enroll/clip — save one enrollment clip (Section 6.5).
 *
 * @param {{signer_id: string, session_id: string, sign: string,
 *          clip: string}} payload  `clip` is base64-encoded clip bytes.
 * @returns {Promise<{signer_id: string, session_id: string, sign: string,
 *          clip_path: string, count: number}>}
 */
export async function enrollClip(payload) {
  const { data } = await api.post("/enroll/clip", payload);
  return data;
}

/**
 * GET /api/enroll/progress/{signerId} — per-sign clip counts for a signer.
 * @param {string} signerId
 * @returns {Promise<{signer_id: string, counts: Object<string, number>,
 *          total: number}>}
 */
export async function enrollProgress(signerId) {
  const { data } = await api.get(`/enroll/progress/${encodeURIComponent(signerId)}`);
  return data;
}

export default api;
