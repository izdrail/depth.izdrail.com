<script setup>
import { onBeforeUnmount, ref } from 'vue'
import DepthViewer from './components/DepthViewer.vue'

const file = ref(null), inputUrl = ref(''), depthUrl = ref(''), error = ref(''), loading = ref(false)
const resolution = ref(256), checkpoint = ref('depth/Log-stage2'), metadata = ref(null)
function choose(selected) {
  const next = selected?.[0]
  if (!next) return
  if (!['image/jpeg', 'image/png'].includes(next.type)) { error.value = 'Choose a JPEG or PNG image.'; return }
  if (inputUrl.value) URL.revokeObjectURL(inputUrl.value)
  file.value = next; inputUrl.value = URL.createObjectURL(next); depthUrl.value = ''; metadata.value = null; error.value = ''
}
function drop(event) { choose(event.dataTransfer.files) }
async function predict() {
  if (!file.value) { error.value = 'Choose an image first.'; return }
  loading.value = true; error.value = ''
  const body = new FormData(); body.append('image', file.value); body.append('resolution', resolution.value); body.append('checkpoint', checkpoint.value); body.append('seed', '2025')
  try {
    const response = await fetch('/api/predict', { method: 'POST', body })
    const json = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(json.detail || `Prediction failed (${response.status})`)
    depthUrl.value = `data:image/png;base64,${json.depth_png_base64}`; metadata.value = json.metadata
  } catch (reason) { error.value = reason.message || 'Prediction failed.' } finally { loading.value = false }
}
function download() { const link = document.createElement('a'); link.href = depthUrl.value; link.download = `depth_${Date.now()}.png`; link.click() }
onBeforeUnmount(() => { if (inputUrl.value) URL.revokeObjectURL(inputUrl.value) })
</script>
<template>
  <main><header><p class="eyebrow">Marigold V2</p><h1>Monocular depth, from one image</h1><p>Run the released depth model on an NVIDIA GPU.</p></header>
    <section class="card">
      <label class="drop" @drop.prevent="drop" @dragover.prevent tabindex="0"> <input type="file" accept="image/jpeg,image/png" @change="choose($event.target.files)"><strong>Drop a JPEG or PNG here</strong><span>or click to choose a file · up to 15 MB</span></label>
      <div class="controls"><label>Resolution<select v-model.number="resolution"><option :value="256">256 × 256</option><option :value="512">512 × 512</option></select></label><label>Checkpoint<select v-model="checkpoint"><option value="depth/Log-stage2">Depth · Log stage 2</option></select></label></div>
      <button class="primary" :disabled="!file || loading" @click="predict"><span v-if="loading" class="spinner"></span>{{ loading ? 'Predicting…' : 'Predict depth' }}</button>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </section>
    <DepthViewer v-if="depthUrl" :input-url="inputUrl" :depth-url="depthUrl" :metadata="metadata" />
    <button v-if="depthUrl" class="download" @click="download">Download PNG</button>
  </main>
</template>
