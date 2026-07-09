import {
  AnalyzeResponse,
  AnalysisListResponse,
  AnalysisPatch,
  CorrectionsUpdate,
  HealthResponse,
  PlantCreate,
  PlantResponse,
  PlantListResponse,
  PlantAnalysisHistory,
} from "./types";
import { supabase } from "../lib/supabase";

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const TIMEOUT_MS = 3 * 60 * 1000;

async function getAuthHeaders(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error("Not authenticated. Please sign in.");
  return { Authorization: `Bearer ${token}` };
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const authHeaders = await getAuthHeaders();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { ...authHeaders, ...(options.headers as Record<string, string> | undefined) },
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timer);
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Analysis is taking too long. Please try again.");
    }
    throw error;
  }
  clearTimeout(timer);

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const ApiClient = {
  async analyzeImage(fileUri: string, plantId?: string): Promise<AnalyzeResponse> {
    const form = new FormData();
    form.append("file", {
      uri: fileUri,
      name: "photo.jpg",
      type: "image/jpeg",
    } as unknown as Blob);

    const query = plantId ? `?plant_id=${encodeURIComponent(plantId)}` : "";
    return request<AnalyzeResponse>(`/api/v1/analyze${query}`, {
      method: "POST",
      body: form,
    });
  },

  async getAnalysis(id: string): Promise<AnalyzeResponse> {
    return request<AnalyzeResponse>(`/api/v1/analyses/${id}`);
  },

  async listAnalyses(limit?: number): Promise<AnalysisListResponse> {
    const query = limit !== undefined ? `?limit=${limit}` : "";
    return request<AnalysisListResponse>(`/api/v1/analyses${query}`);
  },

  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/v1/health");
  },

  async createPlant(body: PlantCreate): Promise<PlantResponse> {
    return request<PlantResponse>("/api/v1/plants", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  },

  async listPlants(name?: string): Promise<PlantListResponse> {
    const query = name ? `?name=${encodeURIComponent(name)}` : "";
    return request<PlantListResponse>(`/api/v1/plants${query}`);
  },

  async listPlantAnalyses(plantId: string): Promise<PlantAnalysisHistory> {
    return request<PlantAnalysisHistory>(`/api/v1/plants/${encodeURIComponent(plantId)}/analyses`);
  },

  async linkAnalysisToPlant(analysisId: string, plantId: string): Promise<void> {
    await request<unknown>(`/api/v1/analyses/${encodeURIComponent(analysisId)}/plant?plant_id=${encodeURIComponent(plantId)}`, {
      method: "PATCH",
    });
  },

  async patchAnalysis(analysisId: string, patch: AnalysisPatch): Promise<AnalyzeResponse> {
    return request<AnalyzeResponse>(`/api/v1/analyses/${encodeURIComponent(analysisId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
  },

  async saveCorrections(analysisId: string, corrections: CorrectionsUpdate): Promise<AnalyzeResponse> {
    return request<AnalyzeResponse>(`/api/v1/analyses/${encodeURIComponent(analysisId)}/corrections`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(corrections),
    });
  },

  async deleteAnalysis(analysisId: string): Promise<void> {
    await request<unknown>(`/api/v1/analyses/${encodeURIComponent(analysisId)}`, {
      method: "DELETE",
    });
  },

  async deletePlant(plantId: string): Promise<void> {
    await request<unknown>(`/api/v1/plants/${encodeURIComponent(plantId)}`, {
      method: "DELETE",
    });
  },
};
