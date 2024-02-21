<script setup lang="ts">
import {FacePosInfo} from "../inference.ts";


const faces = defineProps<{ faces: FacePosInfo[], image_uri: string, uid: string }>()
</script>

<template>
  <dialog :id="'_'+uid" class="modal">
    <div class="modal-box">
      <h3 :id="uid+'-image'">
        aaaaa
        <img :src="image_uri" alt="">
      </h3>
    </div>
    <form method="dialog" class="modal-backdrop">
      <button> close</button>
    </form>
  </dialog>
</template>

<style scoped>

</style>