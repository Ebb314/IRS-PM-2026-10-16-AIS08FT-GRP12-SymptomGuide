import axios from 'axios'
import type { ChatResponse, Symptom } from '../types/chat'

const api = axios.create({ baseURL: import.meta.env.VITE_API_URL ?? '/api', timeout: 30000 })

export async function startChat(message: string): Promise<ChatResponse> {
  const response = await api.post<ChatResponse>('/chat/start', { message })
  return response.data
}

export async function answerFollowUp(
  sessionId: string,
  answer: 'yes' | 'no',
  symptom: Symptom,
): Promise<ChatResponse> {
  const response = await api.post<ChatResponse>('/chat/answer', {
    session_id: sessionId,
    answer,
    symptom,
  })
  return response.data
}
