<script setup lang="ts">
import {InferenceSession, Tensor} from 'onnxruntime-web';

// defineProps<{ img: HTMLImageElement }>();
document.getElementById("input_file")?.addEventListener("change", (event) => {
  console.log(event);

});

</script>

<template>
  <input type="file" id="input_file">
  <h1 class="underline font-bold">aaaaaa</h1>


</template>

<style scoped>

</style>