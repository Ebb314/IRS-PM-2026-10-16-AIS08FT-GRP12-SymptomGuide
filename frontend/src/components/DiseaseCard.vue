<script setup lang="ts">
import type { DiseaseCandidate } from '../types/chat'
defineProps<{ disease: DiseaseCandidate }>()
</script>
<template>
  <article class="result-card">
    <h3>{{ disease.name }}</h3><p>Ranking score: {{ disease.score.toFixed(2) }} <small>(not a probability)</small></p>
    <details><summary>Returned evidence &amp; source</summary>
      <p>{{ disease.evidence_ids.join(', ') || 'No evidence IDs supplied' }}</p><small>{{ disease.source }}</small>
    </details>
  </article>
</template>

