<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import type { ChatMessage } from '../types/chat'
import MessageBubble from './MessageBubble.vue'

const props = defineProps<{ messages: ChatMessage[] }>()
const viewport = ref<HTMLElement>()
watch(() => props.messages.length, async () => {
  await nextTick()
  viewport.value?.scrollTo({ top: viewport.value.scrollHeight, behavior: 'instant' })
})
</script>

<template>
  <div ref="viewport" class="chat-window" aria-live="polite">
    <MessageBubble v-for="message in messages" :key="message.id" :message="message" />
  </div>
</template>
