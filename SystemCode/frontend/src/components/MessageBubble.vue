<script setup lang="ts">
import type { ChatMessage } from '../types/chat'
defineProps<{ message: ChatMessage }>()
</script>

<template>
  <div class="message-row" :class="message.sender">
    <div class="message-label">{{ message.sender === 'assistant' ? 'Guide' : 'You' }}</div>
    <div class="message-bubble">{{ message.text }}</div>
  </div>
</template>
