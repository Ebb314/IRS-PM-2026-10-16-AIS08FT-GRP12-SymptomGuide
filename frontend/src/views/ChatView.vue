<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import ChatInput from '../components/ChatInput.vue'
import ChatWindow from '../components/ChatWindow.vue'
import DiseaseCard from '../components/DiseaseCard.vue'
import DrugCard from '../components/DrugCard.vue'
import FollowUpButtons from '../components/FollowUpButtons.vue'
import AITrace from '../components/AITrace.vue'
import SymptomPanel from '../components/SymptomPanel.vue'
import { useChatStore } from '../stores/chat'
import { vocabulary } from '../api/chat'
import type { Symptom } from '../types/chat'
const store = useChatStore()
const { response, messages, loading, error, awaitingAnswer, pending, answerMode } = storeToRefs(store)
const vocabularyFilter=ref('')
const words = ref<Symptom[]>([])
const elapsed = ref(0)
let timer: ReturnType<typeof setInterval> | undefined
watch(loading, active => {
  clearInterval(timer)
  elapsed.value = 0
  if (active) {
    const started = Date.now()
    timer = setInterval(() => { elapsed.value = Math.floor((Date.now() - started) / 1000) }, 1000)
  }
})
onUnmounted(() => clearInterval(timer))
onMounted(async () => {
  await store.restore()
  try { words.value = await vocabulary() } catch { /* store displays connection errors on interaction */ }
})
</script>

<template>
  <main class="page-shell">
    <header class="topbar">
      <div class="brand-mark">+</div>
      <div><p class="eyebrow">GC1 · Conversation engineering</p><h1>Symptom Guide</h1></div>
      <div class="toolbar">
        <a class="log-link" href="#symptom-log">Symptom log ({{ response?.evidence.length ?? 0 }})</a>
        <el-button v-if="response" :disabled="loading" @click="store.exportSession">Export session</el-button>
        <el-button v-if="response" :disabled="loading" @click="store.reset">Delete &amp; start over</el-button>
      </div>
    </header>
    <section class="notice"><strong>Academic demonstration.</strong> Use fictional inputs. Results are not diagnoses or treatment advice.
      <span v-if="response"> Adapter: {{ response.mode }} · NLP: {{ response.nlp_engine }}.</span>
      <span v-else>Describe how you feel to start. The reasoning adapter in use is shown after your first message.</span>
    </section>
    <div class="workspace">
      <section class="chat-panel" aria-label="Conversation">
        <div class="conversation-heading"><span class="eyebrow">Your conversation</span><span>{{ response ? 'Version ' + response.version : 'New session' }}</span></div>
        <ChatWindow :messages="messages" />
        <div v-if="loading" class="processing"><span role="status">{{ pending?.route === 'correct' ? 'Saving your symptom record…' : 'Processing your request…' }}</span> <span aria-hidden="true">{{ elapsed }}s</span><small v-if="elapsed >= 20"> Local model responses can take longer; your request is still pending.</small></div>
        <div v-if="error" class="error-box" role="alert">
          <p>{{ error }}</p>
          <el-button v-if="pending" :disabled="loading" @click="store.retry">Retry same request</el-button>
          <el-button :disabled="loading" @click="store.restore">Reload session</el-button>
          <el-button :disabled="loading" @click="store.forget">Start new local view</el-button>
        </div>
        <p v-if="response?.next_symptom && response.symptom_question_active !== false" class="question-reason">{{ response.next_symptom.rationale }}</p>
        <div v-if="awaitingAnswer && !loading" class="answer-mode">
          <small>Answering: {{ response?.next_symptom?.kg_name }}</small>
          <el-button v-if="answerMode === 'choice'" :disabled="!!pending" @click="store.describeAnswer">Describe in my own words</el-button>
          <el-button v-else :disabled="!!pending" @click="store.showChoices">Show answer choices</el-button>
        </div>
        <FollowUpButtons v-if="awaitingAnswer && answerMode === 'choice' && !loading" :disabled="!!pending" @answer="store.answer" />
        <ChatInput :disabled="!!pending" :loading="loading" @send="store.send" @composing="store.describeAnswer" />
        <details class="vocabulary"><summary>Supported vocabulary ({{ words.length }})</summary>
          <input v-model="vocabularyFilter" aria-label="Search vocabulary" placeholder="Search symptom names" />
          <p>{{ words.filter(w => w.kg_name.toLowerCase().includes(vocabularyFilter.toLowerCase())).slice(0, 50).map(w => w.kg_name).join(' · ') }}</p><small>Showing up to 50 matches.</small>
          <small>English, limited vocabulary. Unmatched phrases are not evidence that a symptom is absent.</small>
        </details>
      </section>
      <aside id="symptom-log" class="insights-panel" aria-label="Symptom log sidebar">
        <SymptomPanel :evidence="response?.evidence ?? []" :suggestions="response?.pending_evidence ?? []"
          :disabled="loading || !!pending" :version="response?.version" @change="store.correct" />
        <p v-for="warning in response?.warnings ?? []" :key="warning" class="warning">{{ warning }}</p>
        <details v-if="response?.excluded_evidence.length" class="excluded"><summary>Other-person / historical mentions</summary>
          <p v-for="(item, i) in response.excluded_evidence" :key="i">{{ item.original_text }} · {{ item.experiencer }} / {{ item.temporality }}</p>
        </details>
        <template v-if="response">
          <details class="trace"><summary>How this reply was generated</summary><AITrace :trace="response.ai_trace" /></details>
          <details class="adapter-results" :open="response.status === 'COMPLETE'"><summary>Algorithm adapter results ({{ response.mode }})</summary><p class="panel-help">{{ response.mode === 'kg' ? 'Ranked by PrimeKG knowledge-graph reasoning. Candidates are not confirmed conditions.' : 'Integration output, not confirmed conditions. Random mode uses test fixtures.' }}</p>
          <p class="round">Follow-ups {{ response.round }} · {{ response.status.toLowerCase() }}</p>
          <p v-if="response.stop_reason" class="stop-reason">Stop reason: {{ response.stop_reason.split('_').join(' ') }}</p>
          <div v-if="response.candidate_diseases.length" class="section-title"><p class="eyebrow">{{ response.mode === 'kg' ? 'PrimeKG knowledge graph' : response.mode === 'remote' ? 'Adapter output' : 'Integration fixtures only' }}</p><h2>Candidate results</h2></div>
          <DiseaseCard v-for="item in response.candidate_diseases" :key="item.id" :disease="item" :names="Object.fromEntries(response.evidence.map(e => [e.kg_id, e.kg_name]))" />
          <template v-if="response.drug_recommendations.length">
            <div class="section-title"><h2>KG drug relations</h2></div>
            <DrugCard v-for="item in response.drug_recommendations" :key="item.name + item.related_disease" :drug="item" />
          </template>
          </details>
          <details class="trace"><summary>Developer trace &amp; provenance</summary>
            <p>Unknown evidence is excluded from positive / negative recall inputs.</p>
            <pre>{{ JSON.stringify({ timings_ms: response.timings_ms, evidence_history: response.evidence_history }, null, 2) }}</pre>
          </details>
        </template>
      </aside>
    </div>
    <footer>Academic prototype · NLP, dialogue state and full-stack integration · Algorithm adapters are replaceable</footer>
  </main>
</template>



