import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { answerFollowUp, startChat } from '../api/chat'
import type { ChatMessage, ChatResponse } from '../types/chat'

export const useChatStore = defineStore('chat', () => {
  const response = ref<ChatResponse | null>(null)
  const messages = ref<ChatMessage[]>([
    { id: crypto.randomUUID(), sender: 'assistant', text: 'Hello. Tell me which symptoms you are experiencing, including any you do not have.' },
  ])
  const loading = ref(false)
  const error = ref('')
  const awaitingAnswer = computed(() => response.value?.status === 'RUNNING' && !!response.value.next_symptom)

  async function sendInitial(message: string) {
    loading.value = true
    error.value = ''
    messages.value.push({ id: crypto.randomUUID(), sender: 'user', text: message })
    try {
      response.value = await startChat(message)
      messages.value.push({ id: crypto.randomUUID(), sender: 'assistant', text: response.value.message })
    } catch (cause) {
      error.value = 'The service could not process your message. Please check that the backend is running.'
    } finally {
      loading.value = false
    }
  }

  async function answer(answerValue: 'yes' | 'no') {
    if (!response.value?.next_symptom) return
    const symptom = response.value.next_symptom
    loading.value = true
    error.value = ''
    messages.value.push({ id: crypto.randomUUID(), sender: 'user', text: answerValue === 'yes' ? 'Yes' : 'No' })
    try {
      response.value = await answerFollowUp(response.value.session_id, answerValue, symptom)
      messages.value.push({ id: crypto.randomUUID(), sender: 'assistant', text: response.value.message })
    } catch (cause) {
      error.value = 'Your answer could not be saved. Please try again.'
    } finally {
      loading.value = false
    }
  }

  function reset() {
    response.value = null
    error.value = ''
    messages.value = [{ id: crypto.randomUUID(), sender: 'assistant', text: 'Let’s start again. Describe your symptoms.' }]
  }

  return { response, messages, loading, error, awaitingAnswer, sendInitial, answer, reset }
})
