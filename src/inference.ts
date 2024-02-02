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
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
        }).then((session) => {
            retina_face_instance = session;
            finish_counter.value++;
            if (finish_counter.value == 2) visible_loader.value = false;
        })
    });
    console.log(FACE_RECOGNITION_HASH)
}

function resize_image(img: string): Promise<string> {
    const image = document.createElement("img");
    return new Promise((resolve, reject) => {
        image.onload = () => {
            const origImageHeight = image.naturalHeight;
            const origImageWidth = image.naturalWidth;
            let imageWidth;
            let imageHeight;
            if (origImageWidth > origImageHeight) {
                imageWidth = 640;
                imageHeight = 640 * origImageHeight / origImageWidth;
            } else if (origImageWidth < origImageHeight) {
                imageHeight = 640;
                imageWidth = 640 * origImageHeight / origImageWidth;
            } else {
                imageHeight = 640;
                imageWidth = 640;
            }
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = 640;
            const ctx = canvas.getContext("2d")!;
            ctx.drawImage(image, 0, 0, imageHeight, imageWidth);
            resolve(canvas.toDataURL())
        }
        image.onerror = (error) => reject(error)
        image.src = img;
    })

}

export async function predictFacePos(img: string) {
    console.log(img)
    const resized_image = await resize_image(img);
    const picture = await Tensor.fromImage(resized_image, {
        dataType: "float32",
        tensorFormat: "BGR",
        tensorLayout: "NCHW",
        height: 640,
        width: 640,
    });
    console.log(picture.dims);
    const begin = performance.now();
    const resp = await retina_face_instance.run({input: picture});
    console.info(`${performance.now() - begin} m_sec`)
    console.log("bbox", resp["bbox"]["data"])
    console.log("confidence", resp["confidence"]["data"])
    console.log("landmark", resp["landmark"]["data"])
}

