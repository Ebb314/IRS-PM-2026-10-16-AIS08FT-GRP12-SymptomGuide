<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ChatResponse, DrugRecommendation } from '../types/chat'

const props = defineProps<{ modelValue: boolean; response: ChatResponse | null; busy?: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void; (e: 'restart'): void; (e: 'export'): void }>()

const showAll = ref(false)
const visible = computed({ get: () => props.modelValue, set: (value: boolean) => emit('update:modelValue', value) })

// How the interview ended decides the headline: a converged top-3 differential, or an inconclusive list.
const outcome = computed(() => {
  const reason = props.response?.stop_reason
  if (reason === 'candidate_threshold') return {
    kind: 'converged', title: 'Most likely conditions',
    text: 'Based on your answers, the conditions below best match your symptoms.' }
  if (reason === 'no_candidates' || !props.response?.candidate_diseases.length) return {
    kind: 'none', title: 'No matching condition found',
    text: 'None of the conditions in the model matched the symptoms you reported.' }
  return {
    kind: 'inconclusive', title: 'Not enough information for a clear result',
    text: 'The questions did not narrow things down enough. These are only possibilities; please consider seeing a doctor.' }
})

const names = computed<Record<string, string>>(() =>
  Object.fromEntries((props.response?.evidence ?? []).map(e => [e.kg_id, e.kg_name])))
const candidates = computed(() => {
  const all = props.response?.candidate_diseases ?? []
  return showAll.value ? all : all.slice(0, 3)
})
const topScore = computed(() => Math.max(...(props.response?.candidate_diseases ?? []).map(c => c.score), 0.0001))
const drugGroups = computed(() => {
  const groups: Record<string, DrugRecommendation[]> = {}
  for (const drug of props.response?.drug_recommendations ?? []) {
    if (!groups[drug.related_disease]) groups[drug.related_disease] = []
    groups[drug.related_disease].push(drug)
  }
  return Object.entries(groups)
})
const reported = computed(() => {
  const ev = props.response?.evidence ?? []
  return {
    present: ev.filter(e => e.status === 'present').map(e => e.kg_name),
    absent: ev.filter(e => e.status === 'absent').map(e => e.kg_name),
    uncertain: ev.filter(e => e.status === 'uncertain').map(e => e.kg_name),
  }
})
const percent = (score: number) => `${Math.round(score * 100)}%`
</script>

<template>
  <el-dialog v-model="visible" class="result-dialog" width="min(720px, 94vw)" align-center append-to-body
             :close-on-click-modal="false" aria-labelledby="result-title">
    <template #header>
      <div class="result-head">
        <span class="result-badge" :class="outcome.kind">
          {{ outcome.kind === 'converged' ? 'Consultation complete' : outcome.kind === 'none' ? 'No match' : 'Inconclusive' }}
        </span>
        <h2 id="result-title">{{ outcome.title }}</h2>
        <p>{{ outcome.text }}</p>
      </div>
    </template>

    <p class="result-disclaimer"><strong>This is not a diagnosis.</strong> It is generated automatically from the
      symptoms you reported. Please consult a healthcare professional, and seek urgent care for severe or worsening symptoms.</p>

    <section v-if="candidates.length" class="result-section" aria-label="Possible conditions">
      <h3 class="result-section-title">Possible conditions</h3>
      <ol class="result-candidates">
        <li v-for="(item, i) in candidates" :key="item.id" class="result-candidate">
          <div class="result-candidate-row">
            <span class="result-rank">{{ i + 1 }}</span>
            <strong>{{ item.name }}</strong>
            <span class="result-score" :title="'Model score ' + item.score.toFixed(3)">{{ percent(item.score) }}</span>
          </div>
          <div class="result-bar" aria-hidden="true"><span :style="{ width: (100 * item.score / topScore) + '%' }"></span></div>
          <small v-if="item.evidence_ids.length">Supported by: {{ item.evidence_ids.map(id => names[id] ?? id).join(', ') }}</small>
        </li>
      </ol>
      <el-button v-if="(response?.candidate_diseases.length ?? 0) > 3" link type="primary" @click="showAll = !showAll">
        {{ showAll ? 'Show top 3 only' : `Show all ${response?.candidate_diseases.length} candidates` }}
      </el-button>
      <small class="result-note">Scores are model outputs used for ranking, not confirmed probabilities.</small>
    </section>

    <section class="result-section" aria-label="Medication information">
      <h3 class="result-section-title">Related medications</h3>
      <template v-if="drugGroups.length">
        <div v-for="[disease, drugs] in drugGroups" :key="disease" class="result-drug-group">
          <p class="result-drug-disease">For {{ disease }}</p>
          <ul>
            <li v-for="drug in drugs" :key="drug.name">
              <strong>{{ drug.name }}</strong>
              <span class="result-chip">{{ drug.relation }}</span>
            </li>
          </ul>
        </div>
        <small class="result-note">From knowledge-graph drug relations ({{ response?.drug_recommendations[0]?.source }}); drugs
          contraindicated for any listed condition are excluded. Never start, stop or change a medicine without advice
          from a doctor or pharmacist.</small>
      </template>
      <p v-else class="result-empty">
        {{ outcome.kind === 'converged'
          ? 'No drug relations are recorded in the knowledge graph for these conditions.'
          : 'Medication information is only shown when the consultation reaches a clear result.' }}
      </p>
    </section>

    <section class="result-section" aria-label="Your answers">
      <h3 class="result-section-title">What you told us</h3>
      <p v-if="reported.present.length"><span class="result-label">Present</span>{{ reported.present.join(', ') }}</p>
      <p v-if="reported.absent.length"><span class="result-label">Absent</span>{{ reported.absent.join(', ') }}</p>
      <p v-if="reported.uncertain.length"><span class="result-label">Not sure</span>{{ reported.uncertain.join(', ') }}</p>
      <small class="result-note">{{ response?.round ?? 0 }} follow-up question(s) answered. You can correct any answer in the
        Symptom log; the result updates automatically.</small>
    </section>

    <template #footer>
      <div class="result-actions">
        <el-button :disabled="busy" @click="emit('export')">Export session</el-button>
        <el-button :disabled="busy" @click="emit('restart')">Start a new consultation</el-button>
        <el-button type="primary" @click="emit('update:modelValue', false)">Back to conversation</el-button>
      </div>
    </template>
  </el-dialog>
</template>
