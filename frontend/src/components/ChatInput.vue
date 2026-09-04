<script setup lang="ts">
import { ref } from 'vue'

defineProps<{ disabled: boolean; loading: boolean }>()
const emit = defineEmits<{ send: [message: string] }>()
const message = ref('')

function submit() {
  const clean = message.value.trim()
  if (!clean) return
  emit('send', clean)
  message.value = ''
}
</script>

<template>
  <div class="composer">
    <el-input
      v-model="message"
      type="textarea"
      :rows="2"
      resize="none"
      maxlength="2000"
      show-word-limit
      placeholder="For example: I have fever and cough, but no chest pain."
      :disabled="disabled || loading"
      @keydown.ctrl.enter="submit"
    />
    <el-button type="primary" :loading="loading" :disabled="disabled || !message.trim()" @click="submit">
      Send
    </el-button>
  </div>
</template>
