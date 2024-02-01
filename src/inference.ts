import {InferenceSession} from "onnxruntime-web";
import {ref} from "vue";

let face_recognition_instance: InferenceSession;
let retina_face_instance: InferenceSession;
export let visible_loader = ref(false);
export let retina_face_load_progress = ref(0.0)
export let face_recognition_load_progress = ref(0.0)

const sleep = (m_sec: number) => new Promise(resolve => setTimeout(resolve, m_sec))

export async function initModel() {
    visible_loader.value = true;
    let finish_counter = 0;
    fetch("./src/assets/models/face_recognition_sim.onnx").then(async resp => {
        const reader = resp.body!.getReader();
        const total = parseInt(resp.headers.get("Content-Length")!)
        let chunk = 0;
        let modelBuffer = new Uint8Array();
        console.log(total)
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunk += value.length;
            console.log(`face recognition: ${chunk * 100 / total}%`);
            face_recognition_load_progress.value = chunk * 100 / total;
            await sleep(30)
            let mergedArray = new Uint8Array(modelBuffer.length + value.length);
            mergedArray.set(modelBuffer);
            mergedArray.set(value, modelBuffer.length);
            modelBuffer = mergedArray;
        }
        face_recognition_instance = await InferenceSession.create(modelBuffer, {
            executionProviders: ["wasm"]
        })
        finish_counter++;
        if (finish_counter == 2) visible_loader.value = false;
    });
    console.log(FACE_RECOGNITION_HASH)

    fetch("./src/assets/models/retinaface_sim.onnx").then(async resp => {
        const reader = resp.body!.getReader();
        const total = parseInt(resp.headers.get("Content-Length")!)
        let chunk = 0;
        let modelBuffer = new Uint8Array();
        console.log(total)
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunk += value.length;
            console.log(`RetinaFace: ${chunk * 100 / total}%`);
            retina_face_load_progress.value = chunk * 100 / total;
            await sleep(30)
            let mergedArray = new Uint8Array(modelBuffer.length + value.length);
            mergedArray.set(modelBuffer);
            mergedArray.set(value, modelBuffer.length);
            modelBuffer = mergedArray;
        }
        retina_face_instance = await InferenceSession.create(modelBuffer, {
            executionProviders: ["wasm"]
        })

        finish_counter++;
        if (finish_counter == 2) visible_loader.value = false;
    });
    console.log(FACE_RECOGNITION_HASH)
}

export async function inference() {

}

