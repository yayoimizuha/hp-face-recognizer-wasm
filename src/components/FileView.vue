<script setup lang="ts">
import {onBeforeUpdate, onMounted, onUpdated, ref} from "vue";
import {FacePosInfo, inferable, predictFacePos} from "../inference.ts";
import {Md5} from "ts-md5";
import FaceView from "./FaceView.vue";

const props = defineProps<{ file: File }>()
const imageURI = ref("");
const uid = ref("")
const faces = ref<FacePosInfo[]>([]);


const displayImage = () => {
  // if (displayed) return;
  // if (!imageURI.value) return;
  const reader = new FileReader();
  reader.readAsDataURL(props.file);
  // console.info("mounted: " + props.file.name)
  reader.onload = () => {
    imageURI.value = reader.result! as string;
    // console.info("loaded: " + props.file.name)
    // console.log("imageURI.value", imageURI.value)
    // const encoded_data = (new TextEncoder()).encode(imageURI.value);
    // console.log("encoded_data", encoded_data);
    const md5_hasher = new Md5()

    uid.value = md5_hasher.appendStr(imageURI.value).end(false) as string;
    console.log("uid", uid.value);

  }

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
const predict_face_pos = async () => {
  predictFacePos(imageURI.value).then(
      (x) => {
        faces.value = x;
        console.log(faces.value);
      }
  )
}

const showModal = ref(false);

const run_modal = () => {
  eval("_" + uid.value + ".showModal()");
}
</script>

<template>
  <div class="carousel-item">
    <div class="card card-compact w-96 bg-base-100 shadow-xl">
      <figure><img :src="imageURI" :alt="file.name"/></figure>
      <div class="card-body">
        <h2 class="card-title" id="file-name" style="word-break: break-all">{{ file.name }}</h2>
        <div class="card-actions justify-end" :id="uid+'-card'">
          <!--          <a class="btn btn-success " :id="uid+'-view-face-pos'" @click="showModal=true" :href="'#'+uid">view</a>-->
          <button class="btn btn-success" :id="uid+'-view-face-pos'" :disabled="faces.length==0"
                  @click="run_modal">view
          </button>
          <FaceView :image_uri="imageURI" :faces="faces" :uid="uid"/>

          <button class="btn btn-primary" v-if="inferable" @click="predict_face_pos">検出
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>

</style>