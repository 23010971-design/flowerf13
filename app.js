// ============================================================
// YOLO11 OBJECT DETECTION - REALTIME WEB APP
// GitHub Pages + ONNX Runtime Web
// Input: 640 x 640
// ============================================================


// ======================== CONFIG =============================

const MODEL_PATH = "./best.onnx";

const INPUT_SIZE = 320;
const CONF_THRESHOLD = 0.60;
const IOU_THRESHOLD = 0.50;
const INFERENCE_INTERVAL = 100;


// ======================== CLASSES ============================

const CLASS_NAMES = [
    "CHIEU_TIM",
    "CHUOI_DUOC_DO_RUNG",
    "CHUOI_MO_KET",
    "DAM_BUT_CAM",
    "DAM_BUT_VANG",
    "HONG",
    "HUYNH_ANH",
    "MAO_GA_DUOI_CONG",
    "MAU_DON_DO",
    "MAU_DON_TRANG",
    "MAU_DON_VANG",
    "MIA_DO",
    "MO_TRANG",
    "NGOC_BUT"
];

const CLASS_COLORS = [
    "#ffff00",
    "#00c8ff",
    "#ffa500",
    "#00ff00",
    "#c850ff",
    "#ff0000",
    "#0080ff",
    "#ff00ff",
    "#80ff00",
    "#ffffff",
    "#00ffff",
    "#8000ff",
    "#ff8000",
    "#00ff80"
];


// ======================== DOM ================================

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const processingCanvas =
    document.getElementById("processingCanvas");

const processingCtx =
    processingCanvas.getContext("2d", {
        willReadFrequently: true
    });

const statusElement = document.getElementById("status");
const fpsElement = document.getElementById("fps");
const objectsElement = document.getElementById("objects");
const detectionsElement = document.getElementById("detections");

const loadButton = document.getElementById("loadButton");
const cameraButton = document.getElementById("cameraButton");
const stopButton = document.getElementById("stopButton");


// ======================== GLOBAL =============================

let session = null;
let stream = null;
let running = false;
let inferenceBusy = false;
let lastInferenceTime = 0;

let frameCounter = 0;
let lastFpsTime = performance.now();


// ======================== STATUS =============================

function setStatus(message) {
    statusElement.textContent = message;
}


// ======================== LOAD MODEL =========================

async function loadModel() {
    if (session) {
        setStatus("Model already loaded");
        return;
    }

    loadButton.disabled = true;
    setStatus("Loading ONNX model...");

    try {
        if (typeof ort === "undefined") {
            throw new Error(
                "ONNX Runtime Web was not loaded. Check the ort.min.js script."
            );
        }

        // Use WASM for broad browser compatibility.
        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        const response = await fetch(MODEL_PATH);

        if (!response.ok) {
            throw new Error(
                `Cannot download model: HTTP ${response.status}. URL: ${MODEL_PATH}`
            );
        }

        const modelBuffer = await response.arrayBuffer();

        if (modelBuffer.byteLength < 1000) {
            throw new Error(
                "The downloaded ONNX file is unexpectedly small."
            );
        }

        session = await ort.InferenceSession.create(
            modelBuffer,
            {
                executionProviders: ["wasm"],
                graphOptimizationLevel: "all"
            }
        );

        console.log("Model loaded successfully");
        console.log("Input names:", session.inputNames);
        console.log("Output names:", session.outputNames);
        console.log("Input metadata:", session.inputMetadata);
        console.log("Output metadata:", session.outputMetadata);

        setStatus("Model loaded - WASM");

    } catch (error) {
        session = null;

        console.error("MODEL LOADING ERROR:", error);

        setStatus("Model loading failed");

        alert(
            "Cannot load model.\n\n" +
            (error.message || String(error)) +
            "\n\nOpen F12 > Console for details."
        );

    } finally {
        loadButton.disabled = false;
    }
}


// ======================== CAMERA =============================

async function startCamera() {
    if (!session) {
        alert("Please load the model first.");
        return;
    }

    if (running) return;

    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error(
                "Camera API unavailable. Open the site using HTTPS."
            );
        }

        setStatus("Requesting camera permission...");

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: {
                    ideal: "environment"
                },
                width: {
                    ideal: 640
                },
                height: {
                    ideal: 480
                }
            },
            audio: false
        });

        video.srcObject = stream;

        await video.play();

        setupOverlay();

        running = true;
        inferenceBusy = false;
        lastInferenceTime = 0;
        lastFpsTime = performance.now();
        frameCounter = 0;

        setStatus("Camera running");

        requestAnimationFrame(inferenceLoop);

    } catch (error) {
        console.error("CAMERA ERROR:", error);

        setStatus("Camera error");

        alert(
            "Cannot start camera.\n\n" +
            (error.message || String(error)) +
            "\n\nAllow camera permission and use HTTPS."
        );
    }
}


// ======================== STOP CAMERA ========================

function stopCamera() {
    running = false;
    inferenceBusy = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    objectsElement.textContent = "0";
    fpsElement.textContent = "0";
    detectionsElement.textContent = "No detection";

    setStatus(session ? "Model loaded" : "Model not loaded");
}


// ======================== OVERLAY ============================

function setupOverlay() {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
}


// ======================== PREPROCESS =========================

function preprocess() {
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height) {
        throw new Error("Camera frame is not ready.");
    }

    processingCanvas.width = INPUT_SIZE;
    processingCanvas.height = INPUT_SIZE;

    processingCtx.fillStyle = "rgb(114,114,114)";
    processingCtx.fillRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );

    // YOLO letterbox resize
    const scale = Math.min(
        INPUT_SIZE / width,
        INPUT_SIZE / height
    );

    const resizedWidth = Math.round(width * scale);
    const resizedHeight = Math.round(height * scale);

    const offsetX = (INPUT_SIZE - resizedWidth) / 2;
    const offsetY = (INPUT_SIZE - resizedHeight) / 2;

    processingCtx.drawImage(
        video,
        0,
        0,
        width,
        height,
        offsetX,
        offsetY,
        resizedWidth,
        resizedHeight
    );

    const imageData = processingCtx.getImageData(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    ).data;

    const planeSize = INPUT_SIZE * INPUT_SIZE;
    const inputData = new Float32Array(3 * planeSize);

    for (let i = 0; i < planeSize; i++) {
        inputData[i] = imageData[i * 4] / 255;
        inputData[planeSize + i] =
            imageData[i * 4 + 1] / 255;
        inputData[2 * planeSize + i] =
            imageData[i * 4 + 2] / 255;
    }

    const tensor = new ort.Tensor(
        "float32",
        inputData,
        [1, 3, INPUT_SIZE, INPUT_SIZE]
    );

    return {
        tensor,
        scale,
        offsetX,
        offsetY
    };
}


// ======================== INFERENCE ==========================

async function runInference() {
    const prep = preprocess();

    const inputName = session.inputNames[0];

    const feeds = {
        [inputName]: prep.tensor
    };

    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    const output = results[outputName];

    if (!output || !output.dims || output.dims.length !== 3) {
        throw new Error(
            "Unexpected YOLO output shape: " +
            JSON.stringify(output?.dims)
        );
    }

    return {
        output,
        scale: prep.scale,
        offsetX: prep.offsetX,
        offsetY: prep.offsetY
    };
}


// ======================== POSTPROCESS ========================

function postprocess(output, scale, offsetX, offsetY) {
    const data = output.data;
    const dims = output.dims;

    // Supports [1, channels, boxes] and [1, boxes, channels].
    const dim1 = dims[1];
    const dim2 = dims[2];

    let channels;
    let numBoxes;
    let channelFirst;

    // YOLO detection output usually has 4 + numberOfClasses channels.
    if (dim1 <= 256 && dim2 > dim1) {
        channels = dim1;
        numBoxes = dim2;
        channelFirst = true;
    } else if (dim2 <= 256 && dim1 > dim2) {
        channels = dim2;
        numBoxes = dim1;
        channelFirst = false;
    } else {
        throw new Error(
            "Cannot determine YOLO output layout: " +
            JSON.stringify(dims)
        );
    }

    const numClasses = channels - 4;

    if (numClasses <= 0) {
        throw new Error(
            "Invalid YOLO output channel count: " + channels
        );
    }

    if (numClasses !== CLASS_NAMES.length) {
        console.warn(
            `Model outputs ${numClasses} classes, but ` +
            `${CLASS_NAMES.length} class names are configured.`
        );
    }

    function valueAt(channel, box) {
        return channelFirst
            ? data[channel * numBoxes + box]
            : data[box * channels + channel];
    }

    const candidates = [];

    for (let i = 0; i < numBoxes; i++) {
        const cx = valueAt(0, i);
        const cy = valueAt(1, i);
        const w = valueAt(2, i);
        const h = valueAt(3, i);

        let bestScore = -Infinity;
        let bestClass = -1;

        for (let c = 0; c < numClasses; c++) {
            const score = valueAt(4 + c, i);

            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (
            !Number.isFinite(bestScore) ||
            bestScore < CONF_THRESHOLD ||
            bestClass < 0
        ) {
            continue;
        }

        // Convert center x/y and width/height to corners.
        // Standard Ultralytics detection output uses input pixels.
        let x1 = cx - w / 2;
        let y1 = cy - h / 2;
        let x2 = cx + w / 2;
        let y2 = cy + h / 2;

        // Undo letterbox padding and resize.
        x1 = (x1 - offsetX) / scale;
        y1 = (y1 - offsetY) / scale;
        x2 = (x2 - offsetX) / scale;
        y2 = (y2 - offsetY) / scale;

        x1 = Math.max(0, Math.min(video.videoWidth, x1));
        y1 = Math.max(0, Math.min(video.videoHeight, y1));
        x2 = Math.max(0, Math.min(video.videoWidth, x2));
        y2 = Math.max(0, Math.min(video.videoHeight, y2));

        if (x2 <= x1 || y2 <= y1) continue;

        candidates.push({
            x1,
            y1,
            x2,
            y2,
            score: bestScore,
            classId: bestClass
        });
    }

    return nms(candidates, IOU_THRESHOLD);
}


// ======================== IOU ================================

function calculateIoU(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersection = Math.max(0, x2 - x1) *
                         Math.max(0, y2 - y1);

    const areaA = Math.max(0, a.x2 - a.x1) *
                  Math.max(0, a.y2 - a.y1);

    const areaB = Math.max(0, b.x2 - b.x1) *
                  Math.max(0, b.y2 - b.y1);

    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
}


// ======================== NMS ================================

function nms(boxes, threshold) {
    const sorted = [...boxes].sort(
        (a, b) => b.score - a.score
    );

    const selected = [];

    while (sorted.length > 0) {
        const current = sorted.shift();
        selected.push(current);

        for (let i = sorted.length - 1; i >= 0; i--) {
            const candidate = sorted[i];

            // Suppress only boxes belonging to the same class.
            if (
                candidate.classId === current.classId &&
                calculateIoU(current, candidate) >= threshold
            ) {
                sorted.splice(i, 1);
            }
        }
    }

    return selected;
}


// ======================== DRAW ================================

function drawDetections(detections) {
    if (
        overlay.width !== video.videoWidth ||
        overlay.height !== video.videoHeight
    ) {
        setupOverlay();
    }

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    overlayCtx.lineWidth = 3;
    overlayCtx.font = "bold 15px Arial";
    overlayCtx.textBaseline = "top";

    for (const detection of detections) {
        const {
            x1, y1, x2, y2, score, classId
        } = detection;

        const className =
            CLASS_NAMES[classId] ?? `Class ${classId}`;

        const color =
            CLASS_COLORS[classId] ?? "#00ff00";

        const label =
            `${className} ${(score * 100).toFixed(1)}%`;

        const width = x2 - x1;
        const height = y2 - y1;

        overlayCtx.strokeStyle = color;
        overlayCtx.strokeRect(x1, y1, width, height);

        const labelWidth =
            overlayCtx.measureText(label).width + 12;

        const labelY = Math.max(0, y1 - 25);

        overlayCtx.fillStyle = color;
        overlayCtx.fillRect(
            x1,
            labelY,
            labelWidth,
            25
        );

        overlayCtx.fillStyle = "#000000";
        overlayCtx.fillText(
            label,
            x1 + 6,
            labelY + 4
        );
    }

    objectsElement.textContent = String(detections.length);

    if (detections.length === 0) {
        detectionsElement.textContent = "No detection";
        return;
    }

    // Build the detection list safely without injecting HTML.
    detectionsElement.replaceChildren();

    for (const detection of detections) {
        const item = document.createElement("div");
        const name = document.createElement("strong");

        name.textContent =
            CLASS_NAMES[detection.classId] ??
            `Class ${detection.classId}`;

        item.appendChild(document.createTextNode("🌸 "));
        item.appendChild(name);
        item.appendChild(
            document.createTextNode(
                ` - ${(detection.score * 100).toFixed(1)}%`
            )
        );

        detectionsElement.appendChild(item);
    }
}


// ======================== FPS ================================

function updateFPS() {
    frameCounter++;

    const now = performance.now();
    const elapsed = now - lastFpsTime;

    if (elapsed >= 1000) {
        fpsElement.textContent =
            (frameCounter * 1000 / elapsed).toFixed(1);

        frameCounter = 0;
        lastFpsTime = now;
    }
}


// ======================== MAIN LOOP ==========================

async function inferenceLoop(timestamp) {
    if (!running) return;

    if (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        !inferenceBusy &&
        timestamp - lastInferenceTime >= INFERENCE_INTERVAL
    ) {
        inferenceBusy = true;
        lastInferenceTime = timestamp;

        try {
            const result = await runInference();

            const detections = postprocess(
                result.output,
                result.scale,
                result.offsetX,
                result.offsetY
            );

            drawDetections(detections);
            updateFPS();

            setStatus("Camera running");

        } catch (error) {
            console.error("INFERENCE ERROR:", error);
            setStatus("Inference error - see Console");

        } finally {
            inferenceBusy = false;
        }
    }

    if (running) {
        requestAnimationFrame(inferenceLoop);
    }
}


// ======================== BUTTONS ============================

loadButton.addEventListener("click", loadModel);
cameraButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);


// ======================== INITIAL STATUS =====================

setStatus("Ready - load model");