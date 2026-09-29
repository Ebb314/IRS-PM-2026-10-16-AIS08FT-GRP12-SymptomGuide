import axios from 'axios'
import type { ChatResponse, Symptom } from '../types/chat'
const api = axios.create({ baseURL: import.meta.env.VITE_API_URL ?? '/api', timeout: 180000 })
export async function mutate(route: string, payload: Record<string, unknown>) {
  return (await api.post<ChatResponse>('/chat/' + route, payload)).data
}
export async function inspect(sessionId: string) {
  return (await api.get<ChatResponse>('/chat/session/' + encodeURIComponent(sessionId))).data
}
export async function deleteSession(sessionId: string) {
  await api.delete('/chat/session/' + encodeURIComponent(sessionId))
}
export async function vocabulary(): Promise<Symptom[]> { return (await api.get('/vocabulary')).data }
export function errorText(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (error.response?.status === 422) return 'Invalid input. Please check your message and reload the session.'
    return 'The service could not be reached. Your pending request can be retried without saving it twice.'
  }
  return 'Unexpected error. Reload the session before continuing.'
}

