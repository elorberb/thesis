import {
  AnalyzeResponse,
  AnalysisListResponse,
  AnalysisPatch,
  HealthResponse,
  PlantCreate,
  PlantResponse,
  PlantListResponse,
  PlantAnalysisHistory,
} from "./types";

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const TIMEOUT_MS = 3 * 60 * 1000;

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
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
  async analyzeImage(fileUri: string, deviceId: string): Promise<AnalyzeResponse> {
    const form = new FormData();
    form.append("file", {
      uri: fileUri,
      name: "photo.jpg",
      type: "image/jpeg",
    } as unknown as Blob);

    return request<AnalyzeResponse>(`/api/v1/analyze?device_id=${encodeURIComponent(deviceId)}`, {
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
