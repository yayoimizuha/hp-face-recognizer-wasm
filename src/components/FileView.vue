<script setup lang="ts">
import { onBeforeUpdate, onMounted, onUpdated, ref} from "vue";

const props = defineProps<{ file: File }>()
const imageURI = ref("");

const displayImage = () => {
  // if (displayed) return;
  // if (!imageURI.value) return;
  const reader = new FileReader();
  reader.readAsDataURL(props.file);
  // console.info("mounted: " + props.file.name)
  reader.onload = () => {
    imageURI.value = reader.result! as string;
    // console.info("loaded: " + props.file.name)
  }
  // console.log(imageURI.value)
  const encoded_data = (new TextEncoder()).encode(imageURI.value);
  crypto.subtle.digest("SHA-256", encoded_data).then((x) => {
    console.log(Array.from(new Uint8Array(x)).map((b) => b.toString(16).padStart(2, "0")).join(""));
  });
}
onUpdated(() => {
  console.log("onUpdated");
  displayImage();
})
onMounted(() => {
  console.log("onMounted");
  displayImage();
})

onBeforeUpdate(() => {
  // imageURI.value = "";
  // console.info("onBeforeUpdate: " + props.file.name)

})

</script>

<template>
  <div class="carousel-item">
    <div class="card card-compact w-96 bg-base-100 shadow-xl">
      <figure><img :src="imageURI" :alt="file.name"/></figure>
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