import {InferenceSession, Tensor} from "onnxruntime-web";
import {ref} from "vue";

let face_recognition_instance: InferenceSession;
let retina_face_instance: InferenceSession;
export let visible_loader = ref(false);
export let retina_face_load_progress = ref(0.0)
export let face_recognition_load_progress = ref(0.0)

const sleep = (m_sec: number) => new Promise(resolve => setTimeout(resolve, m_sec))
export let finish_counter = ref(0);

export async function initModel() {
    finish_counter.value = 0;
    visible_loader.value = true;
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
        InferenceSession.create(modelBuffer, {
            executionProviders: ["wasm"]
        }).then((session) => {
            face_recognition_instance = session;
            finish_counter.value++;
            if (finish_counter.value == 2) visible_loader.value = false;
        })
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
        InferenceSession.create(modelBuffer, {
            executionProviders: ["wasm"]
        }).then((session) => {
            retina_face_instance = session;
            finish_counter.value++;
            if (finish_counter.value == 2) visible_loader.value = false;
        })
    });
    console.log(FACE_RECOGNITION_HASH)
}

export async function predictFacePos(img: string) {
    console.log(img)
    const picture = await Tensor.fromImage(img, {
        dataType: "float32",
        tensorFormat: "RGB",
        tensorLayout: "NCHW",
        resizedHeight: 640,
        resizedWidth: 640
    });
    console.log(picture.dims)
}

