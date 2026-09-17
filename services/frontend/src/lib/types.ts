/** Kiểu dữ liệu khớp với `services/api/app/schemas.py`. */

export type Severity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface Envelope<T> {
  success: boolean;
  data: T | null;
  error: { code: string; message: string } | null;
  meta: Record<string, unknown> | null;
}

export interface LocationOut {
  id: string;
  name: string;
  area_type: string;
}

export interface Detection {
  class_id: string;
  onset: number;
  offset: number;
  confidence: number;
}

export interface EventSummary {
  event_id: string;
  window_start: string;
  window_end: string;
  location: LocationOut | null;
  caption_vi: string;
  caption_en: string;
  severity: Severity;
  risk_score: number;
  n_detections: number;
}

export interface EventDetail extends EventSummary {
  detections: Detection[];
  grounding_score: number | null;
  model_versions: Record<string, string>;
  has_audio: boolean;
}

export interface Citation {
  event_id: string;
  window_start: string;
  caption_vi: string;
  severity: Severity;
  similarity: number;
}

export interface RagAnswer {
  answer: string;
  citations: Citation[];
  retrieved_events: string[];
  provider: string;
}

export interface UploadResult {
  event_id: string;
  severity: Severity;
  risk_score: number;
  caption_vi: string;
  detections: Detection[];
}

export interface TaxonomyClass {
  id: string;
  vi: string;
  tier: string;
  group: string | null;
}

export interface Taxonomy {
  classes: TaxonomyClass[];
  severities: Record<Severity, string>;
  missing_vi: string[];
}

/** Bản tin đẩy qua WebSocket — xem `app/routers/audio.py::broadcast_event`. */
export interface AlertMessage {
  event_id: string;
  severity: Severity;
  risk_score: number;
  caption_vi: string;
  location_id: string;
  window_start: string;
}
