<script setup>
defineProps({ inputUrl: String, depthUrl: String, metadata: Object })
</script>
<template>
  <section class="results" aria-live="polite">
    <figure><figcaption>Input</figcaption><img :src="inputUrl" alt="Uploaded input"></figure>
    <figure><figcaption>Depth</figcaption><img :src="depthUrl" alt="Generated depth map"></figure>
  </section>
  <p v-if="metadata" class="meta">{{ metadata.resolution }}×{{ metadata.resolution }} inference · {{ metadata.original_width }}×{{ metadata.original_height }} input · {{ metadata.inference_duration_ms }} ms</p>
</template>
