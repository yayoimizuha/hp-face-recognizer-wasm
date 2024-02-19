import {InferenceSession, Tensor, TypedTensor} from "onnxruntime-web";
import {ref} from "vue";
import {post_process} from "../post_process_wasm/pkg";

let face_recognition_instance: InferenceSession;
let retina_face_instance: InferenceSession;
export let visible_loader = ref(false);
export let retina_face_load_progress = ref(0.0)
export let face_recognition_load_progress = ref(0.0)

// const sleep = (m_sec: number) => new Promise(resolve => setTimeout(resolve, m_sec))
export let finish_counter = ref(0);

const square_size = 640;

export async function initModel() {
    finish_counter.value = 0;
    visible_loader.value = true;
    fetch("./src/assets/models/face_recognition_sim.onnx").then(async resp => {
        const reader = resp.body!.getReader();
        const total = parseInt(resp.headers.get("Content-Length")!)
        let chunk = 0;
        let modelBuffer = new Uint8Array();
        console.log(total)
        while (setTimeout(() => true, 500)) {
            const {done, value} = await reader.read();
            if (done) break;
            chunk += value.length;
            console.log(`face recognition: ${chunk * 100 / total}%`);
            face_recognition_load_progress.value = chunk * 100 / total;
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

    fetch("./src/assets/models/retinaface_only_nn_sim.onnx").then(async resp => {
        const reader = resp.body!.getReader();
        const total = parseInt(resp.headers.get("Content-Length")!)
        let chunk = 0;
        let modelBuffer = new Uint8Array();
        console.log(total)
        while (setTimeout(() => true, 500)) {
            const {done, value} = await reader.read();
            if (done) break;
            chunk += value.length;
            console.log(`RetinaFace: ${chunk * 100 / total}%`);
            retina_face_load_progress.value = chunk * 100 / total;
            // await sleep(300)
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

function resize_image(img: string): Promise<[string, number]> {
    const image = document.createElement("img");
    return new Promise((resolve, reject) => {
        image.onload = () => {
            const origImageHeight = image.naturalHeight;
            const origImageWidth = image.naturalWidth;
            let imageWidth;
            let imageHeight;
            let scale;
            if (origImageWidth > origImageHeight) {
                imageHeight = square_size;
                imageWidth = square_size * origImageHeight / origImageWidth;
                scale = origImageHeight / square_size;
            } else if (origImageWidth < origImageHeight) {
                imageWidth = square_size;
                imageHeight = square_size * origImageWidth / origImageHeight;
                scale = origImageWidth / square_size;
            } else {
                imageHeight = square_size;
                imageWidth = square_size;
                scale = origImageHeight / square_size;
            }
            const canvas = document.createElement('canvas');
            // document.getElementById("canvas_view")!.appendChild(canvas);
            canvas.width = canvas.height = square_size;
            const ctx = canvas.getContext("2d")!;
            ctx.drawImage(image, 0, 0, imageHeight, imageWidth);
            resolve([canvas.toDataURL(), scale])
        }
        image.onerror = (error) => reject(error)
        image.src = img;
    })

}

export async function predictFacePos(img: string) {
    // console.log(img)
    const [resized_image, scale] = await resize_image(img);
    const picture = await Tensor.fromImage(resized_image, {
        dataType: "float32",
        tensorFormat: "RGB",
        tensorLayout: "NCHW",
        height: square_size,
        width: square_size,
    }) as TypedTensor<"float32">;
    console.log(picture.dims);
    const begin = performance.now();
    const resp = await retina_face_instance.run({input: picture});
    console.info(`${performance.now() - begin} m_sec`)
    const faces = await post_process(resp["confidence"]["data"] as Float32Array, resp["bbox"]["data"] as Float32Array, resp["landmark"]["data"] as Float32Array, scale, square_size)
    console.log(JSON.parse(faces));
    console.log(face_recognition_instance.inputNames,face_recognition_instance.outputNames)
}

