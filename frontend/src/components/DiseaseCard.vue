<script setup lang="ts">
import type { DiseaseCandidate } from '../types/chat'
defineProps<{ disease: DiseaseCandidate }>()
</script>

<template>
  <article class="result-card">
    <div><span class="eyebrow">Candidate</span><h3>{{ disease.name }}</h3></div>
    <el-progress :percentage="Math.round(disease.score * 100)" :stroke-width="8" />
  </article>
</template>
