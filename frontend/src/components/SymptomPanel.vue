<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Evidence, EvidenceStatus } from '../types/chat'
const props = defineProps<{ evidence: Evidence[]; suggestions: Evidence[]; disabled: boolean; version?: number }>()
const emit = defineEmits<{ change: [id: string, status: EvidenceStatus | 'dismiss'] }>()
const filter = ref('')
const statusFilter = ref<EvidenceStatus | ''>('')
const groups: { status: EvidenceStatus; label: string; description: string }[] = [
  { status: 'present', label: 'Present', description: 'You reported having these symptoms.' },
  { status: 'absent', label: 'Absent', description: 'You reported not having these symptoms.' },
  { status: 'uncertain', label: 'Not sure', description: 'Recorded, but their current status needs clarification.' },
]
const visible = computed(() => props.evidence.filter(e => e.kg_name.toLowerCase().includes(filter.value.trim().toLowerCase())))
const displayedGroups = computed(() => groups.filter(g => statusFilter.value
  ? g.status === statusFilter.value
  : !props.evidence.length || visible.value.some(e => e.status === g.status)))
function change(event: Event, item: Evidence) {
  const select = event.target as HTMLSelectElement
  const status = select.value as EvidenceStatus
  // Keep the last server-confirmed status visible until the transaction succeeds.
  select.value = item.status
  if (status !== item.status) emit('change', item.kg_id, status)
}
</script>
<template>
  <section class="symptom-panel" aria-label="Symptom log">
    <div class="panel-heading"><p class="eyebrow">Your editable record</p><h2>Symptom log</h2>
      <p class="panel-help">Check what was saved and use <strong>Change status</strong> to correct it. These are your reports, not medical diagnoses.</p>
    </div>
    <p class="record-feedback" role="status">{{ disabled ? 'Saving / waiting for the current request…' : version ? 'Saved · session version ' + version : 'No symptoms saved yet' }}</p>
    <label class="status-filter">Show symptoms
      <select v-model="statusFilter" aria-label="Filter symptom log by status">
        <option value="">All statuses ({{ evidence.length }})</option>
        <option v-for="group in groups" :key="group.status" :value="group.status">{{ group.label }} ({{ evidence.filter(e => e.status === group.status).length }})</option>
      </select>
    </label>
    <label v-if="evidence.length" class="symptom-filter">Find a saved symptom<input v-model="filter" type="search" placeholder="Search your records" /></label>
    <p v-if="!evidence.length" class="panel-help">No symptoms are recorded yet. Describe your symptoms in chat; each saved symptom will appear here with its own Change status dropdown. Asking what a word means does not add it to your log.</p>
    <p v-if="evidence.length && !visible.length" class="group-empty">No symptoms match your search.</p>
    <section v-for="group in displayedGroups" :key="group.status" :class="['symptom-group', group.status]" :aria-label="group.label + ' symptoms'">
      <h3>{{ group.label }} <span class="symptom-count">{{ evidence.filter(e => e.status === group.status).length }}</span></h3>
      <p class="group-description">{{ group.description }}</p>
      <p v-if="!visible.some(e => e.status === group.status)" class="group-empty">{{ filter ? 'No matching records.' : 'None recorded.' }}</p>
      <article v-for="item in visible.filter(e => e.status === group.status)" :key="item.kg_id" class="symptom-record" :data-symptom-id="item.kg_id">
        <strong>{{ item.kg_name }}</strong>
        <small v-if="item.duration_text || item.severity_text">{{ item.duration_text }} {{ item.severity_text }}</small>
        <label>Change status
          <select :aria-label="'Status for ' + item.kg_name" :value="item.status" :disabled="disabled" @change="change($event, item)">
            <option value="present">Present</option><option value="absent">Absent</option><option value="uncertain">Not sure</option>
          </select>
        </label>
        <details class="record-source"><summary>Record details</summary><p>“{{ item.original_text }}” · turn {{ item.turn }}</p><p>{{ item.kg_id }} · {{ item.source }}</p></details>
      </article>
    </section>
    <section v-if="suggestions.length" class="suggestion-group" aria-label="Suggested matches">
      <h3>Confirm a suggested match <span class="symptom-count">{{ suggestions.length }}</span></h3>
      <p class="group-description">These interpretations are not saved symptoms. If the name matches what you meant, choose its status. Otherwise ignore it.</p>
      <article v-for="item in suggestions" :key="item.kg_id" class="symptom-record">
        <strong>{{ item.kg_name }}</strong><p>Possible meaning of “{{ item.original_text }}”</p>
        <div class="suggestion-actions"><button v-for="group in groups" :key="group.status" :disabled="disabled" :aria-label="'Confirm ' + item.kg_name + ' as ' + group.label" @click="emit('change', item.kg_id, group.status)">{{ group.label }}</button>
          <button :disabled="disabled" @click="emit('change', item.kg_id, 'dismiss')">Ignore suggestion</button></div>
      </article>
    </section>
  </section>
</template>
