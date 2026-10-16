<script setup lang="ts">
defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ answer: [value: 'yes' | 'no' | 'unknown'] }>()
</script>
<template>
  <div class="answer-buttons" aria-label="Answer the follow-up question">
    <el-button type="primary" :disabled="disabled" @click="emit('answer', 'yes')">Yes</el-button>
    <el-button :disabled="disabled" @click="emit('answer', 'no')">No</el-button>
    <el-button :disabled="disabled" @click="emit('answer', 'unknown')">Not sure</el-button>
  </div>
</template>

