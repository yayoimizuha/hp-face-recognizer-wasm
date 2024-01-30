<!--suppress ES6UnusedImports -->
<script setup lang="ts">


import FileView from "./components/FileView.vue";
import {ref} from "vue";

let files = ref<Array<File>>([]);

function updateFile() {
  // files.value = [];
  // // document.getElementById("fileList")!.innerHTML="";
  // for (const _file of (document.getElementById("filePicker")! as HTMLInputElement).files!) {
  //   console.log(_file)
  //   files.value.push(_file)
  // }
  files.value = Array.from((document.getElementById("filePicker")! as HTMLInputElement).files!);
  files.value.map((x) => console.log(x));
}
</script>

<template>
  <div class="navbar bg-base-100">
    <a class="btn btn-ghost text-lg sm:text-xl"> Hello!Projectメンバーの顔識別アプリ・WASM版 </a>
  </div>

  <div class="card bg-secondary-content max-w-xl m-auto">
    <div class="card-body">
      <a>
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        RetinaFaceとファインチューニング済みFaceNetによる顔認識アプリです。<br>
        ファイルはデバイス内で処理され、外部には送信されません。<br>
        『誤認識を報告。』ボタンを押すと、サーバーに保存されるのでご注意ください。 </a>
    </div>

  </div>
  <div class="divider divider-secondary"></div>
  <div class="m-auto flex justify-center gap-4">
    <input type="file" multiple class="file-input file-input-bordered file-input-primary max-w-xs w-full"
           id="filePicker" @change="updateFile"/>
    <button class="btn btn-secondary">全て検出</button>
  </div>

  <div class="divider divider-info"></div>

  <div class="justify-center flex">
    <div id="fileList"
         class="carousel carousel-center max-w-[calc(100%-32px)] lg:max-w-[calc(1024px-32px)] p-4 m-4 space-x-4 bg-neutral rounded-box max-h-[50vh]">
      <FileView v-for="file in files" :file="file"></FileView>
    </div>
  </div>

  <!--  <HelloWorld msg="Vite + Vue"/>-->
  <!--  <Inference/>-->
</template>

<!--<style scoped>-->
<!--.logo {-->
<!--  height: 6em;-->
<!--  padding: 1.5em;-->
<!--  will-change: filter;-->
<!--  transition: filter 300ms;-->
<!--}-->

<!--.logo:hover {-->
<!--  filter: drop-shadow(0 0 2em #646cffaa);-->
<!--}-->

<!--.logo.vue:hover {-->
<!--  filter: drop-shadow(0 0 2em #42b883aa);-->
<!--}-->
<!--</style>-->
