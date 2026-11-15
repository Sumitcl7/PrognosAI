export interface ApiHealth {
  status: string;
  model: string;
  features: number;
  drift_monitor: string;
}

export interface FleetEvaluation {
  evaluation: {
    dataset: string;
    cohort: string;
    mode: string;
    total_engines: number;
    labels_used_for_prediction: boolean;
  };

  summary: {
    risk_counts: Record<string, number>;
    drift_counts: Record<string, number>;
    maintenance_counts: Record<string, number>;
    maintenance_alerts: number;
    model_reviews_required: number;
    average_predicted_rul: number;
  };

  engines: EnginePrediction[];
}

export interface EnginePrediction {
  engine_id: number;
  latest_cycle: number;
  history_observations: number;
  predicted_stage: string;
  predicted_rul: number;
  lower_bound: number;
  upper_bound: number;
  interval_width: number;
  confidence_level: number;
  risk_level: string;
  confidence_category: string;
  maintenance_action: string;
  model_review_required: boolean;
  drift_status: string;
  drift_evaluated: boolean;
}

const API_BASE = "/api";

async function request<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`);

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`API request failed (${response.status}): ${message}`);
  }

  return response.json() as Promise<T>;
}

export function getApiHealth(): Promise<ApiHealth> {
  return request<ApiHealth>("/health");
}

export function getFleetEvaluation(): Promise<FleetEvaluation> {
  return request<FleetEvaluation>("/fleet/evaluation");
}

export interface EngineHistoryPoint {
  cycle: number;
  predicted_rul: number;
  predicted_stage: string;
  observation_interval: {
    lower_bound: number;
    upper_bound: number;
    width: number;
  };
  engine_interval: {
    lower_bound: number;
    upper_bound: number;
    width: number;
  };
}

export interface EngineHistory {
  engine_id: number;
  dataset: string;
  evaluation_mode: string;
  model: string;
  history_observations: number;
  confidence_level: number;
  labels_used_for_prediction: boolean;
  predictions: EngineHistoryPoint[];
}

export function getEngineHistory(engineId: number): Promise<EngineHistory> {
  return request<EngineHistory>(`/engines/${engineId}/history`);
}
