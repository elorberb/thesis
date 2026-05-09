export type TrichomeType = "clear" | "cloudy" | "amber";

export type MaturityStage = "early" | "developing" | "peak" | "mature" | "late";

export type BoundingBox = {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
};

export type Detection = {
  bbox: BoundingBox;
  trichome_type: TrichomeType;
  confidence: number;
};

export type TrichomeResult = {
  detections: Detection[];
  distribution: Record<TrichomeType, number>;
  total_count: number;
};

export type StigmaDetection = {
  bbox: BoundingBox;
  green_ratio: number;
  orange_ratio: number;
};

export type StigmaResult = {
  detections: StigmaDetection[];
  avg_green_ratio: number;
  avg_orange_ratio: number;
  total_count: number;
};

export type AnalyzeResponse = {
  id: string;
  created_at: string;
  device_id: string;
  plant_id: string | null;
  image_url: string;
  annotated_image_url: string | null;
  trichome_result: TrichomeResult;
  stigma_result: StigmaResult;
  maturity_stage: MaturityStage;
  recommendation: string;
  trichome_crops_b64: string[] | null;
  stigma_crops_b64: string[] | null;
};

export type AnalysisListItem = {
  id: string;
  created_at: string;
  device_id: string;
  plant_id: string | null;
  image_url: string;
  annotated_image_url: string | null;
  maturity_stage: MaturityStage;
  recommendation: string;
  trichome_distribution: Record<TrichomeType, number> | null;
  stigma_ratios: { green: number; orange: number } | null;
};

export type AnalysisListResponse = {
  items: AnalysisListItem[];
  total: number;
};

export type PlantCreate = {
  name: string;
  metadata?: Record<string, string>;
};

export type PlantResponse = {
  id: string;
  created_by: string | null;
  name: string;
  status: "active" | "harvested" | "removed";
  metadata: Record<string, string>;
  created_at: string;
};

export type PlantListResponse = {
  items: PlantResponse[];
  total: number;
};

export type PlantAnalysisItem = {
  id: string;
  created_at: string;
  device_id: string;
  plant_id: string | null;
  image_url: string;
  annotated_image_url: string | null;
  maturity_stage: MaturityStage;
  recommendation: string;
  trichome_distribution: Record<TrichomeType, number> | null;
  stigma_ratios: { green: number; orange: number } | null;
};

export type PlantAnalysisHistory = {
  plant_id: string;
  items: PlantAnalysisItem[];
  total: number;
};

export type AnalysisPatch = {
  maturity_stage?: MaturityStage;
  recommendation?: string;
  trichome_distribution?: Record<TrichomeType, number>;
  stigma_ratios?: { green: number; orange: number };
};

export type HealthResponse = {
  status: string;
};
