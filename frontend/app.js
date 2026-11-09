/**
 * SignConnect - Frontend Application
 * Real-time MediaPipe Hand Tracking + FastAPI Sign Transformer Inference
 */

(() => {
  // ---------------- DOM Elements ----------------
  const video = document.getElementById('webcamVideo');
  const canvas = document.getElementById('outputCanvas');
  const ctx = canvas.getContext('2d');
  const viewportOverlay = document.getElementById('viewportPlaceholder');
  const startCameraBtn = document.getElementById('startCameraBtn');
  const toggleCameraBtn = document.getElementById('toggleCameraBtn');
  const toggleCameraText = document.getElementById('toggleCameraText');
  const flipCameraBtn = document.getElementById('flipCameraBtn');
  const handsPrompt = document.getElementById('handsPrompt');
  const handTrackingPill = document.getElementById('handTrackingPill');
  const fpsVal = document.getElementById('fpsVal');

  const bufferCount = document.getElementById('bufferCount');
  const bufferFill = document.getElementById('bufferFill');

  const backendStatus = document.getElementById('backendStatus');
  const vocabToggle = document.getElementById('vocabToggle');
  const vocabTags = document.getElementById('vocabTags');

  const predictedSignText = document.getElementById('predictedSignText');
  const inferenceSubtext = document.getElementById('inferenceSubtext');
  const confidenceTag = document.getElementById('confidenceTag');
  const confidenceBar = document.getElementById('confidenceBar');
  const confidencePercent = document.getElementById('confidencePercent');
  const candidatesList = document.getElementById('candidatesList');
  const speakSignBtn = document.getElementById('speakSignBtn');
  const commitSignBtn = document.getElementById('commitSignBtn');

  const sentenceDisplay = document.getElementById('sentenceDisplay');
  const tokensPreview = document.getElementById('tokensPreview');
  const modeNlpBtn = document.getElementById('modeNlpBtn');
  const modeAiBtn = document.getElementById('modeAiBtn');
  const modeRawBtn = document.getElementById('modeRawBtn');
  const sentenceMethodBadge = document.getElementById('sentenceMethodBadge');
  const targetLanguage = document.getElementById('targetLanguage');
  const aiKeyConfigBtn = document.getElementById('aiKeyConfigBtn');
  const apiKeyPanel = document.getElementById('apiKeyPanel');
  const groqApiKeyInput = document.getElementById('groqApiKeyInput');
  const saveApiKeyBtn = document.getElementById('saveApiKeyBtn');
  const clearApiKeyBtn = document.getElementById('clearApiKeyBtn');
  const predictSentenceBtn = document.getElementById('predictSentenceBtn');

  const addSpaceBtn = document.getElementById('addSpaceBtn');
  const backspaceBtn = document.getElementById('backspaceBtn');
  const clearSentenceBtn = document.getElementById('clearSentenceBtn');
  const copySentenceBtn = document.getElementById('copySentenceBtn');
  const speakSentenceBtn = document.getElementById('speakSentenceBtn');
  const autoCommitToggle = document.getElementById('autoCommitToggle');

  const historyList = document.getElementById('historyList');
  const emptyHistoryPrompt = document.getElementById('emptyHistoryPrompt');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  const toastContainer = document.getElementById('toastContainer');

  // ---------------- State Variables ----------------
  const FRAMES = 30;
  const FEATURES = 126;
  const CONFIDENCE_THRESHOLD = 0.70;
  const AUTO_COMMIT_THRESHOLD = 0.75;
  const INFERENCE_STRIDE_MS = 250; // Max 4 inferences per second

  let isCameraRunning = false;
  let isMirrored = true;
  let cameraStream = null;
  let cameraInstance = null;
  let handsModel = null;

  let sequenceBuffer = [];
  let noHandFramesCount = 0;
  let lastInferenceTime = 0;
  let isInferenceInFlight = false;

  let lastRecognizedSign = "";
  let lastRecognizedConf = 0;
  let lastCommittedSign = "";
  let lastCommitTime = 0;

  let sentenceTokens = [];
  let sentenceMode = "nlp"; // 'nlp', 'ai_model', or 'raw'
  let savedGroqApiKey = localStorage.getItem('signconnect_groq_api_key') || "";
  let supportedClasses = [];

  let frameCounter = 0;
  let lastFpsUpdateTime = performance.now();

  // Hand Connections index pairs for drawing skeleton
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],         // Index
    [5, 9], [9, 10], [10, 11], [11, 12],     // Middle
    [9, 13], [13, 14], [14, 15], [15, 16],   // Ring
    [13, 17], [17, 18], [18, 19], [19, 20],  // Pinky
    [0, 17]                                  // Palm base
  ];

  // ---------------- Initial Setup ----------------
  init();

  async function init() {
    setupEventListeners();
    await checkBackendHealth();
    await fetchSupportedClasses();
    setInterval(checkBackendHealth, 8000);
  }

  function setupEventListeners() {
    startCameraBtn.addEventListener('click', startCamera);
    toggleCameraBtn.addEventListener('click', toggleCamera);
    flipCameraBtn.addEventListener('click', toggleMirror);

    vocabToggle.addEventListener('click', () => {
      vocabTags.classList.toggle('collapsed');
    });

    modeNlpBtn.addEventListener('click', () => setSentenceMode('nlp'));
    modeAiBtn.addEventListener('click', () => setSentenceMode('ai_model'));
    modeRawBtn.addEventListener('click', () => setSentenceMode('raw'));

    targetLanguage.addEventListener('change', () => {
      syncSentenceWithBackend();
    });

    aiKeyConfigBtn.addEventListener('click', () => {
      apiKeyPanel.classList.toggle('hidden');
      if (!apiKeyPanel.classList.contains('hidden') && savedGroqApiKey) {
        groqApiKeyInput.value = savedGroqApiKey;
      }
    });

    saveApiKeyBtn.addEventListener('click', () => {
      const key = groqApiKeyInput.value.trim();
      savedGroqApiKey = key;
      localStorage.setItem('signconnect_groq_api_key', key);
      apiKeyPanel.classList.add('hidden');
      showToast("Groq API Key saved successfully!", "success");
      if (sentenceMode === 'ai_model') {
        syncSentenceWithBackend();
      }
    });

    clearApiKeyBtn.addEventListener('click', () => {
      savedGroqApiKey = "";
      groqApiKeyInput.value = "";
      localStorage.removeItem('signconnect_groq_api_key');
      apiKeyPanel.classList.add('hidden');
      showToast("API Key cleared. Using local NLP engine.", "info");
      syncSentenceWithBackend();
    });

    predictSentenceBtn.addEventListener('click', () => {
      if (sentenceTokens.length === 0) {
        showToast("Add or perform some signs first to predict a sentence.", "info");
        return;
      }
      syncSentenceWithBackend(true);
    });

    commitSignBtn.addEventListener('click', () => {
      if (lastRecognizedSign && lastRecognizedSign !== "READY") {
        commitWord(lastRecognizedSign);
      }
    });

    speakSignBtn.addEventListener('click', () => {
      if (lastRecognizedSign && lastRecognizedSign !== "READY") {
        speakText(lastRecognizedSign);
      }
    });

    addSpaceBtn.addEventListener('click', () => {
      sentenceTokens.push(" ");
      updateSentenceUI();
    });

    backspaceBtn.addEventListener('click', () => {
      if (sentenceTokens.length > 0) {
        sentenceTokens.pop();
        syncSentenceWithBackend();
      }
    });

    clearSentenceBtn.addEventListener('click', () => {
      sentenceTokens = [];
      sentenceDisplay.value = "";
      renderTokenChips();
    });

    copySentenceBtn.addEventListener('click', () => {
      const text = sentenceDisplay.value.trim();
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        showToast("Sentence copied to clipboard!", "success");
      }).catch(() => {
        showToast("Failed to copy sentence", "error");
      });
    });

    speakSentenceBtn.addEventListener('click', () => {
      const text = sentenceDisplay.value.trim();
      if (text) speakText(text);
    });

    clearHistoryBtn.addEventListener('click', () => {
      historyList.innerHTML = '';
      historyList.appendChild(emptyHistoryPrompt);
      emptyHistoryPrompt.style.display = 'block';
    });
  }

  // ---------------- Backend Communication ----------------
  async function checkBackendHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const dot = backendStatus.querySelector('.status-dot');
        const label = backendStatus.querySelector('.status-label');
        dot.className = 'status-dot online';
        label.textContent = `Online (${data.backend})`;
      } else {
        setBackendOffline();
      }
    } catch {
      setBackendOffline();
    }
  }

  function setBackendOffline() {
    const dot = backendStatus.querySelector('.status-dot');
    const label = backendStatus.querySelector('.status-label');
    dot.className = 'status-dot offline';
    label.textContent = 'Backend Offline';
  }

  async function fetchSupportedClasses() {
    try {
      const res = await fetch('/classes');
      if (res.ok) {
        const data = await res.json();
        supportedClasses = data.classes || [];
        renderVocabTags(supportedClasses);
      }
    } catch (e) {
      console.warn("Could not load classes:", e);
    }
  }

  function renderVocabTags(classes) {
    vocabTags.innerHTML = '';
    classes.forEach(cls => {
      const chip = document.createElement('span');
      chip.className = 'vocab-chip';
      chip.id = `vocab-${cls}`;
      chip.textContent = cls;
      vocabTags.appendChild(chip);
    });
  }

  function highlightVocabTag(sign) {
    document.querySelectorAll('.vocab-chip').forEach(c => c.classList.remove('highlight'));
    const target = document.getElementById(`vocab-${sign}`);
    if (target) {
      target.classList.add('highlight');
    }
  }

  // ---------------- Camera & MediaPipe ----------------
  async function toggleCamera() {
    if (isCameraRunning) {
      stopCamera();
    } else {
      await startCamera();
    }
  }

  function toggleMirror() {
    isMirrored = !isMirrored;
    if (isMirrored) {
      video.classList.add('mirrored');
      canvas.classList.add('mirrored');
    } else {
      video.classList.remove('mirrored');
      canvas.classList.remove('mirrored');
    }
  }

  async function startCamera() {
    try {
      showToast("Initializing camera...", "info");
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      cameraStream = stream;
      video.srcObject = stream;
      await video.play();

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      if (isMirrored) {
        video.classList.add('mirrored');
        canvas.classList.add('mirrored');
      }

      // Initialize MediaPipe Hands if not already done
      if (!handsModel) {
        await initMediaPipeHands();
      }

      isCameraRunning = true;
      viewportOverlay.classList.add('hidden');
      toggleCameraText.textContent = "Stop Camera";
      toggleCameraBtn.classList.replace('btn-primary', 'btn-danger');

      setTrackingStatus("no_hands");
      requestAnimationFrame(processVideoFrame);
      showToast("Camera started successfully", "success");
    } catch (err) {
      console.error("Camera error:", err);
      let msg = "Could not access camera. Please check permissions.";
      if (err.name === 'NotAllowedError') {
        msg = "Camera permission was denied. Please grant permission in browser settings.";
      } else if (err.name === 'NotFoundError') {
        msg = "No webcam device was found on this system.";
      }
      showToast(msg, "error");
    }
  }

  function stopCamera() {
    isCameraRunning = false;
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    viewportOverlay.classList.remove('hidden');
    handsPrompt.classList.add('hidden');
    toggleCameraText.textContent = "Start Camera";
    toggleCameraBtn.classList.replace('btn-danger', 'btn-primary');
    
    sequenceBuffer = [];
    updateBufferUI(0);
    setTrackingStatus("off");
    fpsVal.textContent = "0";
  }

  async function initMediaPipeHands() {
    if (typeof Hands === 'undefined') {
      throw new Error("MediaPipe Hands library not loaded from CDN.");
    }

    handsModel = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    handsModel.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    handsModel.onResults(onHandResults);
  }

  async function processVideoFrame() {
    if (!isCameraRunning) return;

    // Calculate FPS
    frameCounter++;
    const now = performance.now();
    if (now - lastFpsUpdateTime >= 1000) {
      fpsVal.textContent = frameCounter.toString();
      frameCounter = 0;
      lastFpsUpdateTime = now;
    }

    if (handsModel && video.readyState >= 2) {
      await handsModel.send({ image: video });
    }

    if (isCameraRunning) {
      requestAnimationFrame(processVideoFrame);
    }
  }

  // ---------------- Hand Landmarks Extraction & Normalization ----------------
  /**
   * Exact normalization matching the sign_transformer model training:
   * 1. 21 points (x, y, z) per hand = 63 coords.
   * 2. Centered relative to wrist: pts -= pts[0].
   * 3. Normalized by L2 norm: pts /= (norm(pts) + 1e-6).
   * 4. Up to 2 hands: hand 1 at 0..62, hand 2 at 63..125.
   * Total 126 features per frame.
   */
  function extractHandFeatures(multiHandLandmarks) {
    if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
      return null;
    }

    const features = new Float32Array(FEATURES); // initialized to 0.0
    let offset = 0;
    const handsToProcess = multiHandLandmarks.slice(0, 2);

    for (const landmarks of handsToProcess) {
      const wrist = landmarks[0];
      let sumSq = 0;
      const centered = [];

      for (let i = 0; i < 21; i++) {
        const dx = landmarks[i].x - wrist.x;
        const dy = landmarks[i].y - wrist.y;
        const dz = (landmarks[i].z !== undefined ? landmarks[i].z : 0) - (wrist.z !== undefined ? wrist.z : 0);
        centered.push(dx, dy, dz);
        sumSq += dx * dx + dy * dy + dz * dz;
      }

      const norm = Math.sqrt(sumSq) + 1e-6;

      for (let i = 0; i < 63; i++) {
        features[offset + i] = centered[i] / norm;
      }

      offset += 63;
    }

    return Array.from(features);
  }

  function onHandResults(results) {
    // 1. Draw landmarks on overlay canvas
    drawHandLandmarks(results);

    // 2. Extract features
    const handCount = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;
    
    if (handCount > 0) {
      noHandFramesCount = 0;
      handsPrompt.classList.add('hidden');
      setTrackingStatus(handCount === 1 ? "1_hand" : "2_hands");

      const frameFeatures = extractHandFeatures(results.multiHandLandmarks);
      if (frameFeatures) {
        sequenceBuffer.push(frameFeatures);
        if (sequenceBuffer.length > FRAMES) {
          sequenceBuffer.shift();
        }
      }
    } else {
      noHandFramesCount++;
      setTrackingStatus("no_hands");
      // If hands missing for 6 frames, clear the buffer
      if (noHandFramesCount >= 6) {
        handsPrompt.classList.remove('hidden');
        if (sequenceBuffer.length > 0) {
          sequenceBuffer = [];
        }
      }
    }

    updateBufferUI(sequenceBuffer.length);

    // 3. Trigger inference when 30 frames are ready
    const now = performance.now();
    if (
      sequenceBuffer.length === FRAMES &&
      !isInferenceInFlight &&
      (now - lastInferenceTime >= INFERENCE_STRIDE_MS)
    ) {
      lastInferenceTime = now;
      executeInference(sequenceBuffer);
    }
  }

  // ---------------- Canvas Landmark Visualizer ----------------
  function drawHandLandmarks(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return;

    for (let h = 0; h < results.multiHandLandmarks.length; h++) {
      const landmarks = results.multiHandLandmarks[h];
      const isFirstHand = h === 0;
      const boneColor = isFirstHand ? 'rgba(6, 182, 212, 0.7)' : 'rgba(139, 92, 246, 0.7)';
      const jointColor = isFirstHand ? '#38bdf8' : '#a78bfa';

      // Draw bones
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = boneColor;
      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = landmarks[startIdx];
        const p2 = landmarks[endIdx];
        ctx.beginPath();
        ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
        ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
        ctx.stroke();
      }

      // Draw joints
      for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        const x = pt.x * canvas.width;
        const y = pt.y * canvas.height;
        const radius = (i === 0 || i === 4 || i === 8 || i === 12 || i === 16 || i === 20) ? 5 : 3.5;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = jointColor;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  // ---------------- Model Inference ----------------
  async function executeInference(buffer) {
    isInferenceInFlight = true;
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: buffer })
      });

      if (!response.ok) {
        throw new Error(`Inference HTTP ${response.status}`);
      }

      const result = await response.json();
      handleInferenceResult(result);
    } catch (err) {
      console.error("Inference request failed:", err);
    } finally {
      isInferenceInFlight = false;
    }
  }

  function handleInferenceResult(result) {
    const sign = result.sign;
    const confidence = result.confidence;
    const candidates = result.top_candidates || [];

    lastRecognizedSign = sign;
    lastRecognizedConf = confidence;

    // 1. Update Sign Text & Animation
    predictedSignText.textContent = sign;
    predictedSignText.classList.remove('pop');
    void predictedSignText.offsetWidth; // trigger reflow
    predictedSignText.classList.add('pop');

    inferenceSubtext.textContent = `Model inferred with ${(confidence * 100).toFixed(1)}% certainty`;

    // 2. Update Confidence Meter
    const pct = Math.round(confidence * 100);
    confidencePercent.textContent = `${pct}%`;
    confidenceTag.textContent = `Confidence: ${pct}%`;
    confidenceBar.style.width = `${pct}%`;

    if (confidence >= 0.80) {
      confidenceBar.style.background = 'linear-gradient(90deg, #10b981, #06b6d4)';
    } else if (confidence >= 0.60) {
      confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #10b981)';
    } else {
      confidenceBar.style.background = 'linear-gradient(90deg, #f43f5e, #f59e0b)';
    }

    // 3. Render Top Candidates
    renderCandidates(candidates);

    // 4. Highlight vocabulary tag
    highlightVocabTag(sign);

    // 5. Append to History Log
    addHistoryItem(sign, confidence);

    // 6. Auto-commit to sentence if high confidence & debounce
    if (autoCommitToggle.checked && confidence >= AUTO_COMMIT_THRESHOLD) {
      const now = performance.now();
      const timeSinceLastCommit = now - lastCommitTime;
      // Auto-commit if different sign OR same sign after 2.5s pause
      if (sign !== lastCommittedSign || timeSinceLastCommit > 2500) {
        lastCommittedSign = sign;
        lastCommitTime = now;
        commitWord(sign);
      }
    }
  }

  function renderCandidates(candidates) {
    candidatesList.innerHTML = '';
    candidates.forEach(cand => {
      const item = document.createElement('div');
      item.className = 'candidate-item';
      const pct = Math.round(cand.confidence * 100);
      item.innerHTML = `
        <span class="candidate-name">${cand.sign}</span>
        <div class="candidate-track">
          <div class="candidate-fill" style="width: ${pct}%"></div>
        </div>
        <span class="candidate-val">${pct}%</span>
      `;
      candidatesList.appendChild(item);
    });
  }

  // ---------------- Sentence Builder ----------------
  function commitWord(word) {
    sentenceTokens.push(word);
    renderTokenChips();
    syncSentenceWithBackend();
    showToast(`Added: ${word}`, "info");
  }

  function setSentenceMode(mode) {
    sentenceMode = mode;
    modeNlpBtn.classList.toggle('active', mode === 'nlp');
    modeAiBtn.classList.toggle('active', mode === 'ai_model');
    modeRawBtn.classList.toggle('active', mode === 'raw');

    if (mode === 'ai_model' && !savedGroqApiKey) {
      showToast("Tip: Click 'AI Key' to provide a Groq key for cloud LLaMA 3, or enjoy automatic local NLP fallback.", "info");
    }

    syncSentenceWithBackend();
  }

  async function syncSentenceWithBackend(userTriggered = false) {
    renderTokenChips();
    if (sentenceTokens.length === 0) {
      sentenceDisplay.value = "";
      updateMethodBadge("empty");
      return;
    }

    if (userTriggered) {
      predictSentenceBtn.disabled = true;
      predictSentenceBtn.innerHTML = `Predicting...`;
    }

    try {
      const res = await fetch('/sentence/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          words: sentenceTokens,
          mode: sentenceMode,
          language: targetLanguage.value,
          api_key: savedGroqApiKey || null
        })
      });

      if (res.ok) {
        const data = await res.json();
        sentenceDisplay.value = data.sentence || "";
        updateMethodBadge(data.method, data.notice);
        if (userTriggered) {
          showToast(`Predicted: "${data.sentence}"`, "success");
        }
      } else {
        sentenceDisplay.value = sentenceTokens.join(" ");
        updateMethodBadge("fallback");
      }
    } catch {
      sentenceDisplay.value = sentenceTokens.join(" ");
      updateMethodBadge("offline");
    } finally {
      if (userTriggered) {
        predictSentenceBtn.disabled = false;
        predictSentenceBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          Predict Sentence
        `;
      }
    }
  }

  function updateMethodBadge(method, notice) {
    if (!sentenceMethodBadge) return;
    if (method === "llama_3.1_generative_ai") {
      sentenceMethodBadge.textContent = "🧠 LLaMA 3.1 AI Model";
      sentenceMethodBadge.style.color = "var(--accent-purple)";
      sentenceMethodBadge.style.borderColor = "rgba(139, 92, 246, 0.4)";
      sentenceMethodBadge.style.background = "rgba(139, 92, 246, 0.12)";
    } else if (method === "computational_nlp_engine") {
      sentenceMethodBadge.textContent = "⚡ NLP Engine Active";
      sentenceMethodBadge.style.color = "var(--accent-cyan)";
      sentenceMethodBadge.style.borderColor = "rgba(6, 182, 212, 0.3)";
      sentenceMethodBadge.style.background = "rgba(6, 182, 212, 0.12)";
    } else if (method === "local_nlp_fallback") {
      sentenceMethodBadge.textContent = "⚡ NLP Engine (AI Fallback)";
      sentenceMethodBadge.style.color = "var(--accent-amber)";
      sentenceMethodBadge.style.borderColor = "rgba(245, 158, 11, 0.3)";
      sentenceMethodBadge.style.background = "rgba(245, 158, 11, 0.12)";
    } else if (method === "raw_concatenation") {
      sentenceMethodBadge.textContent = "🔤 Raw Sign Gloss";
      sentenceMethodBadge.style.color = "var(--text-muted)";
      sentenceMethodBadge.style.borderColor = "var(--border-color)";
      sentenceMethodBadge.style.background = "rgba(255, 255, 255, 0.04)";
    }
  }

  function renderTokenChips() {
    tokensPreview.innerHTML = '';
    sentenceTokens.forEach((token, index) => {
      const chip = document.createElement('span');
      chip.className = 'token-chip';
      chip.innerHTML = `
        <span>${token}</span>
        <span class="token-delete" title="Remove">&times;</span>
      `;
      chip.querySelector('.token-delete').addEventListener('click', () => {
        sentenceTokens.splice(index, 1);
        syncSentenceWithBackend();
      });
      tokensPreview.appendChild(chip);
    });
  }

  // ---------------- Recognition History ----------------
  function addHistoryItem(sign, confidence) {
    if (emptyHistoryPrompt) {
      emptyHistoryPrompt.style.display = 'none';
    }

    const item = document.createElement('div');
    item.className = 'history-item';
    
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const pct = Math.round(confidence * 100);

    item.innerHTML = `
      <span class="history-sign">${sign}</span>
      <div class="history-meta">
        <span class="history-conf">${pct}%</span>
        <span>${timeStr}</span>
      </div>
    `;

    // Click history item to add to sentence
    item.addEventListener('click', () => {
      commitWord(sign);
    });

    historyList.prepend(item);

    // Limit history list to 50 items
    while (historyList.children.length > 50) {
      historyList.removeChild(historyList.lastChild);
    }
  }

  // ---------------- Text to Speech ----------------
  function speakText(text) {
    if (!('speechSynthesis' in window)) {
      showToast("Text-to-speech not supported in this browser.", "error");
      return;
    }

    window.speechSynthesis.cancel(); // Stop any pending speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.lang = 'en-US';

    window.speechSynthesis.speak(utterance);
    showToast(`Speaking: "${text}"`, "info");
  }

  // ---------------- UI Helpers ----------------
  function updateBufferUI(count) {
    bufferCount.textContent = `${count} / ${FRAMES} frames`;
    const pct = Math.min(100, Math.round((count / FRAMES) * 100));
    bufferFill.style.width = `${pct}%`;
  }

  function setTrackingStatus(status) {
    const dot = handTrackingPill.querySelector('.pill-dot');
    const text = handTrackingPill.querySelector('.pill-text');

    switch (status) {
      case 'off':
        dot.className = 'pill-dot gray';
        text.textContent = 'Camera Off';
        break;
      case 'no_hands':
        dot.className = 'pill-dot red';
        text.textContent = 'No Hands Detected';
        break;
      case '1_hand':
        dot.className = 'pill-dot yellow';
        text.textContent = '1 Hand Tracked';
        break;
      case '2_hands':
        dot.className = 'pill-dot green';
        text.textContent = '2 Hands Tracked';
        break;
    }
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }
})();
