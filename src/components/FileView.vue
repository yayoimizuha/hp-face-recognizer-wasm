<script setup lang="ts">
import {onMounted, ref} from "vue";

const props = defineProps<{ file: File }>()
const imageURI = ref("");

onMounted(() => {
  const reader = new FileReader();
  reader.readAsDataURL(props.file);
  reader.onload = () => {
    imageURI.value = reader.result! as string;
  }
})

</script>

<template>
  <div class="carousel-item">
    <div class="card card-compact w-96 bg-base-100 shadow-xl">
      <figure><img :src="imageURI" alt="test 1"/></figure>
      <div class="card-body">
        <h2 class="card-title" id="file-name">{{ file.name }}</h2>
        <!--          <p>If a dog chews shoes whose shoes does he choose?</p>-->
        <div class="card-actions justify-end">
          <button class="btn btn-primary">検出</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>

</style>