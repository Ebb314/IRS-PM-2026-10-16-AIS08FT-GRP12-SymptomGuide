<script setup lang="ts">
defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ answer: [value: 'yes' | 'no'] }>()
</script>

<template>
  <div class="answer-buttons" aria-label="Answer the follow-up question">
    <el-button type="primary" size="large" :disabled="disabled" @click="emit('answer', 'yes')">Yes</el-button>
    <el-button size="large" :disabled="disabled" @click="emit('answer', 'no')">No</el-button>
  </div>
</template>
