<script setup lang="ts">
defineProps<{ trace: any }>()
</script>
<template>
  <section v-if="trace && Object.keys(trace).length" class="ai-trace-card">
    <p class="eyebrow">AI execution</p>
    <template v-if="trace.engine === 'conversation-agent-v1'">
      <p>Conversation: <strong>contextual LLM</strong></p>
      <p v-if="trace.understanding?.interpretation?.user_goal">Understood goal: {{ trace.understanding.interpretation.user_goal }}</p>
      <p v-if="trace.understanding?.interpretation?.unresolved">Still unclear: {{ trace.understanding.interpretation.unresolved }}</p>
      <p>Evidence: {{ trace.understanding?.validation }}</p>
    </template>
    <p v-else>Intent: <strong>{{ trace.intent || 'report' }}</strong></p>
    <p v-if="trace.routing">Routing: {{ trace.routing.mode }}<span v-if="trace.routing.confidence != null"> · model confidence {{ trace.routing.confidence }}</span></p>
    <p v-if="trace.response_mode">Reply: {{ trace.response_mode }}</p>
    <p v-if="trace.response_mode === 'local-llm-general'">AI explanation from model knowledge; no retrieved sources.</p>
    <p v-if="trace.nlu?.validation">Understanding: {{ trace.nlu.validation }}</p>
    <p v-if="trace.generation?.response_mode">Question: {{ trace.generation.response_mode }}</p>
    <p v-if="trace.routing?.consistency_check">Intent consistency: {{ trace.routing.consistency_check.verdict }}</p>
    <div v-for="(call, i) in [trace.understanding?.model_call, trace.understanding?.repair_call, trace.routing?.model_call, trace.routing?.consistency_check?.model_call, trace.nlu?.model_call, trace.model_call, trace.generation?.model_call].filter(Boolean)" :key="i" class="model-call">
      <strong>{{ call.model }}</strong><small>{{ call.device }} · {{ call.output_tokens }} output tokens · {{ call.latency_ms }} ms</small>
    </div>
    <div v-if="trace.citations?.length" class="source-list"><h3>Answer sources</h3>
      <article v-for="source in trace.citations" :key="source.id">
        <a v-if="source.url" :href="source.url" target="_blank" rel="noopener noreferrer">{{ source.title }}</a>
        <strong v-else>{{ source.title }}</strong><p>{{ source.text }}</p>
      </article>
    </div>
    <details v-if="trace.understanding?.candidates?.length || trace.nlu?.candidates?.length || trace.retrieval?.length"><summary>Retrieval ranking</summary>
      <p v-for="hit in (trace.understanding?.candidates || trace.nlu?.candidates || trace.retrieval)" :key="hit.id">{{ hit.title }} · BM25 {{ hit.bm25 }} · cosine {{ hit.cosine }} · RRF {{ hit.rrf }}</p>
    </details>
  </section>
</template>

