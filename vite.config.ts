import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import {readFileSync} from "fs";
import {viteStaticCopy} from "vite-plugin-static-copy";

async function gen_hash(file_path: string) {
    const file_content = readFileSync(file_path);
    const hashArray = crypto.subtle.digest("SHA-256", file_content)
    const uint8_view = new Uint8Array(await hashArray);
    const hashString = Array.from(uint8_view).map((c) => c.toString(16).padStart(2, "0")).join("");
    console.log(hashString);
    return JSON.stringify(hashString)
}

// https://vitejs.dev/config/
export default defineConfig(async () => {
    const retina_face_hash = await gen_hash("./src/assets/models/retinaface_sim.onnx");
    const face_recognition_hash = await gen_hash("./src/assets/models/face_recognition_sim.onnx");
    return {
        plugins: [vue(),
            viteStaticCopy({
                targets: [
                    {
                        src: "./node_modules/onnxruntime-web/dist/*.wasm",
                        dest: "./",
                    }
                ]
            })],
        define:
            {
                RETINA_FACE_HASH: retina_face_hash,
                FACE_RECOGNITION_HASH: face_recognition_hash
            }
    }
})
