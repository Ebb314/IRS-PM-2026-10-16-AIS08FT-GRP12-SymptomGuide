export interface Symptom { kg_id: string; kg_name: string }
export type EvidenceStatus = 'present' | 'absent' | 'uncertain'
export interface Evidence extends Symptom {
  status: EvidenceStatus; original_text: string; turn: number; source: string;
  experiencer: string; temporality: string; confidence: number; start: number; end: number; duration_text: string; severity_text: string
}
export interface NextSymptom extends Symptom { heuristic_score: number; rationale: string }
export interface DiseaseCandidate { id: string; name: string; score: number; evidence_ids: string[]; source: string }
export interface DrugRecommendation { name: string; related_disease: string; note: string; relation: string; source: string }
export interface ChatMessage { id: string; sender: 'user' | 'assistant'; text: string }
export interface ChatResponse {
  symptom_question_active?: boolean;
  pending_evidence?: Evidence[];

  session_id: string; status: 'RUNNING' | 'COMPLETE'; message: string;
  positive_symptoms: Symptom[]; negative_symptoms: Symptom[]; uncertain_symptoms: Symptom[];
  evidence: Evidence[]; evidence_history: Evidence[]; excluded_evidence: Evidence[];
  candidate_diseases: DiseaseCandidate[]; next_symptom: NextSymptom | null;
  drug_recommendations: DrugRecommendation[]; round: number; version: number;
  stop_reason: string | null; warnings: string[]; messages: ChatMessage[];
  timings_ms: Record<string, number>; mode: string; nlp_engine: string; ai_trace: Record<string, any>
}


