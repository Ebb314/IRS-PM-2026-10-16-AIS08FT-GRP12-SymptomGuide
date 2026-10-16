<script setup lang="ts">
import type { DiseaseCandidate } from '../types/chat'
defineProps<{ disease: DiseaseCandidate; names?: Record<string, string> }>()
</script>
<template>
  <article class="result-card">
    <h3>{{ disease.name }}</h3><p>Ranking score: {{ disease.score.toFixed(2) }} <small>(not a probability)</small></p>
    <details><summary>Supported by &amp; source</summary>
      <p>{{ disease.evidence_ids.map(id => names?.[id] ?? id).join(', ') || 'No evidence IDs supplied' }}</p><small>{{ disease.source }}</small>
    </details>
  </article>
</template>

