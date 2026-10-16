import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { mutate, inspect, deleteSession, errorText } from '../api/chat'
import type { ChatResponse, EvidenceStatus } from '../types/chat'

export const useChatStore = defineStore('chat', () => {
  const response = ref<ChatResponse | null>(null)
  const loading = ref(false)
  const error = ref('')
  const answerMode = ref<'choice' | 'text'>('choice')
  const pending = ref<{ route: string; payload: Record<string, unknown> } | null>(null)
  const messages = computed(() => response.value?.messages ?? [
    { id:'welcome', sender:'assistant' as const, text:'Tell me what is on your mind. You can describe how you feel, ask what a health term means, or ask for help finding the right words.' },
  ])
  const awaitingAnswer = computed(() => response.value?.symptom_question_active !== false && response.value?.status === 'RUNNING' && !!response.value.next_symptom)
  function accept(value: ChatResponse) {
    if (value.next_symptom?.kg_id !== response.value?.next_symptom?.kg_id) answerMode.value='choice'
    response.value = value
    sessionStorage.setItem('symptom-session', value.session_id)
  }
  function base() {
    return { session_id: response.value!.session_id, expected_version: response.value!.version, request_id: crypto.randomUUID() }
  }
  async function retry() {
    if (!pending.value || loading.value) return
    loading.value = true; error.value = ''
    try {
      accept(await mutate(pending.value.route, pending.value.payload))
      pending.value = null
    } catch (cause) { error.value = errorText(cause) }
    finally { loading.value = false }
  }
  async function send(message: string) {
    if (loading.value || pending.value) return
    pending.value = response.value
      ? { route:'message', payload:{ ...base(), message } }
      : { route:'start', payload:{ message, request_id:crypto.randomUUID() } }
    await retry()
  }
  async function answer(answer: 'yes' | 'no' | 'unknown') {
    if (!response.value?.next_symptom || loading.value || pending.value) return
    pending.value = { route:'answer', payload:{ ...base(), answer, symptom:response.value.next_symptom && {
      kg_id:response.value.next_symptom.kg_id, kg_name:response.value.next_symptom.kg_name,
    } } }
    await retry()
  }
  async function correct(kg_id: string, status: EvidenceStatus | 'dismiss') {
    if (!response.value || loading.value || pending.value) return
    pending.value = { route:'correct', payload:{ ...base(), kg_id, status } }
    await retry()
  }
  async function restore() {
    const id = response.value?.session_id ?? sessionStorage.getItem('symptom-session')
    if (!id || loading.value) return
    loading.value = true; error.value = ''
    try { accept(await inspect(id)); pending.value = null }
    catch (cause) { error.value = errorText(cause) }
    finally { loading.value = false }
  }
  async function reset() {
    if (loading.value) return
    loading.value = true
    try {
      const id = response.value?.session_id ?? sessionStorage.getItem('symptom-session')
      if (id) await deleteSession(id)
      response.value = null; pending.value = null; error.value = ''
      sessionStorage.removeItem('symptom-session')
    } catch (cause) { error.value = errorText(cause) }
    finally { loading.value = false }
  }
  function forget() {
    // Local recovery if a session was already deleted or the backend DB was reset.
    if (loading.value) return
    response.value = null; pending.value = null; error.value = ''
    sessionStorage.removeItem('symptom-session')
  }
  function exportSession() {
    if (!response.value) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(response.value, null, 2)], {type:'application/json'}))
    const a = document.createElement('a'); a.href = url; a.download = 'dialogue-session.json'; a.click()
    URL.revokeObjectURL(url)
  }
  function describeAnswer() { answerMode.value='text' }
  function showChoices() { answerMode.value='choice' }
  return {response, loading, error, pending, messages, awaitingAnswer, answerMode, describeAnswer, showChoices, send, answer, correct, retry, restore, reset, forget, exportSession}
})

