export interface Symptom {
  kg_id: string
  kg_name: string
}

export interface NextSymptom extends Symptom {
  heuristic_score: number
}

export interface DiseaseCandidate {
  id: string
  name: string
  score: number
}

export interface DrugRecommendation {
  name: string
  related_disease: string
  note?: string
}

export interface ChatResponse {
  session_id: string
  status: 'RUNNING' | 'COMPLETE'
  message: string
  positive_symptoms: Symptom[]
  negative_symptoms: Symptom[]
  candidate_diseases: DiseaseCandidate[]
  next_symptom: NextSymptom | null
  drug_recommendations: DrugRecommendation[]
  round: number
}

export interface ChatMessage {
  id: string
  sender: 'user' | 'assistant'
  text: string
}
