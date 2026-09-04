<script setup lang="ts">
import { storeToRefs } from 'pinia'
import ChatInput from '../components/ChatInput.vue'
import ChatWindow from '../components/ChatWindow.vue'
import DiseaseCard from '../components/DiseaseCard.vue'
import DrugCard from '../components/DrugCard.vue'
import FollowUpButtons from '../components/FollowUpButtons.vue'
import { useChatStore } from '../stores/chat'

const store = useChatStore()
const { response, messages, loading, error, awaitingAnswer } = storeToRefs(store)
</script>

<template>
  <main class="page-shell">
    <header class="topbar">
      <div class="brand-mark">+</div>
      <div><p class="eyebrow">Knowledge graph prototype</p><h1>Symptom Guide</h1></div>
      <el-button v-if="response" text @click="store.reset">Start over</el-button>
    </header>

    <section class="notice"><strong>Educational prototype</strong> — not a diagnosis or a substitute for professional medical care. Seek emergency help for severe symptoms.</section>

    <div class="workspace">
      <section class="chat-panel">
        <ChatWindow :messages="messages" />
        <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />
        <FollowUpButtons v-if="awaitingAnswer" :disabled="loading" @answer="store.answer" />
        <ChatInput v-else-if="!response || (response.status === 'RUNNING' && !response.next_symptom)" :disabled="false" :loading="loading" @send="store.sendInitial" />
      </section>

      <aside class="insights-panel">
        <div class="panel-heading"><p class="eyebrow">Session evidence</p><h2>What we know</h2></div>
        <div v-if="response" class="evidence">
          <div><span>Present</span><el-tag v-for="item in response.positive_symptoms" :key="item.kg_id" type="success">{{ item.kg_name }}</el-tag></div>
          <div><span>Absent</span><el-tag v-for="item in response.negative_symptoms" :key="item.kg_id" type="info">{{ item.kg_name }}</el-tag></div>
          <p class="round">Round {{ response.round }} · {{ response.status.toLowerCase() }}</p>
        </div>
        <p v-else class="empty-state">Your structured symptom evidence and candidate results will appear here.</p>

        <template v-if="response?.status === 'COMPLETE'">
          <div class="section-title"><p class="eyebrow">Results</p><h2>Disease candidates</h2></div>
          <DiseaseCard v-for="item in response.candidate_diseases" :key="item.id" :disease="item" />
          <div class="section-title"><p class="eyebrow">Treatment layer</p><h2>Drug output</h2></div>
          <DrugCard v-for="item in response.drug_recommendations" :key="`${item.name}-${item.related_disease}`" :drug="item" />
        </template>
      </aside>
    </div>
  </main>
</template>
