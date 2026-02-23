// ---- Config ----
// Put your exported ONNX here:
const MODEL_URL = "models/mrnet_abnormal_sagittal.onnx";

// Class labels for a binary classifier.
// You can change these to match TASK='acl' or 'meniscus' or 'abnormal'.
const NEG_LABEL = "negative (0)";
const POS_LABEL = "positive (1)";

// ImageNet normalization (must match your training transform)
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

let session = null;
let loadedImage = null;

const el = (id) => document.getElementById(id);

function setStatus(msg) {
  el("status").textContent = `Status: ${msg}`;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Resize + preprocess into Float32 NCHW [1, 3, 224, 224]
function imageToTensor(img) {
  const size = 224;

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  // Draw image resized to 224x224
  ctx.drawImage(img, 0, 0, size, size);

  const { data } = ctx.getImageData(0, 0, size, size); // RGBA uint8

  // Create NCHW float32 tensor
  const floatData = new Float32Array(1 * 3 * size * size);

  // data is [r,g,b,a, r,g,b,a, ...]
  // Convert to [0..1], normalize, then store NCHW
  const HW = size * size;
  for (let i = 0; i < HW; i++) {
    const r = data[i * 4 + 0] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    // Normalize
    const rn = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    const gn = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    const bn = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

    // NCHW indexing
    floatData[0 * 3 * HW + 0 * HW + i] = rn;
    floatData[0 * 3 * HW + 1 * HW + i] = gn;
    floatData[0 * 3 * HW + 2 * HW + i] = bn;
  }

  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

async function loadModel() {
  setStatus("loading model...");
  el("loadBtn").disabled = true;
  try {
    // Some models run faster with wasm; ORT auto-selects.
    session = await ort.InferenceSession.create(MODEL_URL);
    setStatus("model loaded ✅");
    el("runBtn").disabled = !loadedImage;
  } catch (err) {
    console.error(err);
    setStatus(`model load failed: ${err.message || err}`);
    el("loadBtn").disabled = false;
  }
}

async function runInference() {
  if (!session) {
    setStatus("please load the model first");
    return;
  }
  if (!loadedImage) {
    setStatus("please drop an image first");
    return;
  }

  setStatus("running inference...");
  el("runBtn").disabled = true;

  try {
    const inputTensor = imageToTensor(loadedImage);

    // Your ONNX export names: input -> "input", output -> "logits"
    // If they differ, log session.inputNames / outputNames.
    const feeds = {};
    const inputName = session.inputNames[0];  // safest
    feeds[inputName] = inputTensor;

    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    const logitsTensor = results[outputName];

    const logits = logitsTensor.data[0];          // scalar
    const prob = sigmoid(logits);
    const pred = prob > 0.5 ? 1 : 0;

    el("logits").textContent = logits.toFixed(6);
    el("prob").textContent = prob.toFixed(6);
    el("pred").textContent = pred === 1 ? `${POS_LABEL}` : `${NEG_LABEL}`;

    setStatus("done ✅");
  } catch (err) {
    console.error(err);
    setStatus(`inference failed: ${err.message || err}`);
  } finally {
    el("runBtn").disabled = false;
  }
}

function setPreviewFromFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    loadedImage = img;

    el("preview").src = url;
    el("preview").style.display = "block";
    el("noPreview").style.display = "none";

    el("runBtn").disabled = !session;
    setStatus("image loaded");
  };
  img.onerror = () => {
    setStatus("failed to load image");
  };
  img.src = url;
}

function setupDropzone() {
  const dropzone = el("dropzone");
  const fileInput = el("fileInput");

  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) setPreviewFromFile(file);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) setPreviewFromFile(file);
  });
}

// Init
el("modelPathLabel").textContent = MODEL_URL;
el("loadBtn").addEventListener("click", loadModel);
el("runBtn").addEventListener("click", runInference);
setupDropzone();
setStatus("ready (drop an image, then load model)");
