<script lang="ts">
    import {InferenceSession} from "onnxruntime-web";

    const InferenceOption = {executionProviders: ['wasm']};
    // const InferenceOption = {executionProviders: ['wasm', 'webgl']};
    let session = InferenceSession.create("/face_recognition_sim.onnx", InferenceOption);
</script>
<div class="inference">

</div>