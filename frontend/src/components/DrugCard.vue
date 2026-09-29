<script setup lang="ts">
import type { DrugRecommendation } from '../types/chat'
defineProps<{ drug: DrugRecommendation }>()
</script>
<template>
  <article class="result-card drug-card">
    <span class="eyebrow">Knowledge graph relation</span><h3>{{ drug.name }}</h3>
    <p>{{ drug.relation }} · {{ drug.related_disease }}</p><p>{{ drug.note }}</p>
    <small>Source: {{ drug.source }}. This relation is not an individual treatment recommendation.</small>
  </article>
</template>

