import {InferenceSession, Tensor, TypedTensor} from "onnxruntime-web";
import {ref} from "vue";
import {post_process} from "../post_process_wasm/pkg";

let face_recognition_instance: InferenceSession;
let retina_face_instance: InferenceSession;
export let visible_loader = ref(false);
export let retina_face_load_progress = ref(0.0)
export let face_recognition_load_progress = ref(0.0)

export let inferable = ref(false);

// const sleep = (m_sec: number) => new Promise(resolve => setTimeout(resolve, m_sec))
export let finish_counter = ref(0);

const square_size = 640;

export type FacePosInfo = {
    bbox: number[],
    score: number,
    landmarks: number[][]
}

export async function initModel() {
    const model_load_button = document.getElementById("model_load_button")! as HTMLButtonElement;
    model_load_button.disabled = true;
    finish_counter.value = 0;
    visible_loader.value = true;
    const face_recognition_promise = new Promise((resolve, reject) => {
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
                    resolve("");
                }, err => reject(err))
            },
            err => reject(err));
    })

    console.log(FACE_RECOGNITION_HASH)
    const retina_face_promise = new Promise((resolve, reject) => {
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
                resolve("")
            }, err => reject(err))
        }, err => reject(err));
    });
    Promise.all([retina_face_promise, face_recognition_promise]).then(
        () => {
            inferable.value = true;
            model_load_button.classList.remove("btn-secondary");
            model_load_button.classList.add("btn-success");
            document.getElementById("model_load_button_text")!.innerText = "読み込み完了";
            setTimeout(() => visible_loader.value = false, 1400)
        },
        () => model_load_button.disabled = false
    )

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

export async function predictFacePos(img: string, uid: string) {
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
    post_process(resp["confidence"]["data"] as Float32Array, resp["bbox"]["data"] as Float32Array, resp["landmark"]["data"] as Float32Array, scale, square_size).then(
        (post_process_res) => {
            const faces = JSON.parse(post_process_res) as FacePosInfo[]
            if (faces.length != 0) {
                const button_area = document.getElementById(uid)! as HTMLDivElement;
                button_area.prepend(
                    new HTMLButtonElement()
                )
            }
            console.log(faces);
            console.log(face_recognition_instance.inputNames, face_recognition_instance.outputNames)
        }
    )
}

