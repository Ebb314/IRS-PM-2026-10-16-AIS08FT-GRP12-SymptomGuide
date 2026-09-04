<script setup lang="ts">
import type { DrugRecommendation } from '../types/chat'
defineProps<{ drug: DrugRecommendation }>()
</script>

<template>
  <article class="result-card drug-card">
    <span class="eyebrow">Mock recommendation</span>
    <h3>{{ drug.name }}</h3>
    <p>Related to {{ drug.related_disease }}</p>
    <small v-if="drug.note">{{ drug.note }}</small>
  </article>
</template>
