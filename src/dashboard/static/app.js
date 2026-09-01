let currentSelectedFeature = 0;
let currentActiveLayer = "";

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initLayers();
  initSearch();
  initSteeringControls();
  initAnalyze();
});

function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

async function initLayers() {
  const selectEl = document.getElementById("layer-select");
  try {
    const res = await fetch("/api/layers");
    if (!res.ok) return;
    const data = await res.json();
    currentActiveLayer = data.active_layer || (data.layers.length > 0 ? data.layers[0] : "");

    selectEl.innerHTML = data.layers.map(l => `
      <option value="${l}" ${l === currentActiveLayer ? "selected" : ""}>${l}</option>
    `).join("");

    selectEl.addEventListener("change", async (e) => {
      const newLayer = e.target.value;
      const changeRes = await fetch("/api/select_layer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hook_point: newLayer })
      });
      if (changeRes.ok) {
        currentActiveLayer = newLayer;
        loadFeatures();
      }
    });

    loadFeatures();
  } catch (err) {
    console.error("Could not initialize layers:", err);
  }
}

function initSearch() {
  const searchInput = document.getElementById("feature-search");
  let debounceTimer;
  searchInput.addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      loadFeatures(e.target.value);
    }, 300);
  });
}

async function loadFeatures(searchQuery = "") {
  const listEl = document.getElementById("feature-list");
  listEl.innerHTML = '<div class="loading">Loading features...</div>';
  try {
    let url = `/api/features?page=1&page_size=100&layer=${encodeURIComponent(currentActiveLayer)}`;
    if (searchQuery) {
      url += `&search=${encodeURIComponent(searchQuery)}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error("Backend not initialized or no model loaded.");
    const data = await res.json();
    document.getElementById("total-count").innerText = `${data.total_features} Features`;
    listEl.innerHTML = "";

    if (data.features.length === 0) {
      listEl.innerHTML = '<div class="feature-item"><p style="color:#71717a">No matching features found.</p></div>';
      return;
    }

    data.features.forEach((feat, idx) => {
      const item = document.createElement("div");
      item.className = `feature-item ${idx === 0 ? "active" : ""}`;
      item.innerHTML = `
        <h4>Feature #${feat.feature_id}</h4>
        <p>${feat.explanation}</p>
      `;
      item.addEventListener("click", () => {
        document.querySelectorAll(".feature-item").forEach(i => i.classList.remove("active"));
        item.classList.add("active");
        selectFeature(feat.feature_id, feat.explanation);
      });
      listEl.appendChild(item);
    });

    if (data.features.length > 0) {
      selectFeature(data.features[0].feature_id, data.features[0].explanation);
    }
  } catch (err) {
    listEl.innerHTML = `<div class="feature-item"><p style="color:#ef4444">Could not load features: ${err.message}</p></div>`;
  }
}

async function selectFeature(featureId, explanation) {
  currentSelectedFeature = featureId;
  const titleEl = document.getElementById("inspect-title");
  const expEl = document.getElementById("inspect-explanation");
  const steerFeatInput = document.getElementById("steer-feat-id");

  if (titleEl) titleEl.innerText = `Feature #${featureId} (${currentActiveLayer})`;
  if (expEl) expEl.innerText = explanation || `Feature #${featureId}`;
  if (steerFeatInput) steerFeatInput.value = featureId;

  // Fetch Logit Lens Attribution
  try {
    const res = await fetch(`/api/feature/${featureId}/logits?top_k=10&layer=${encodeURIComponent(currentActiveLayer)}`);
    if (!res.ok) return;
    const data = await res.json();

    const promList = document.getElementById("promoted-list");
    if (promList) {
      promList.innerHTML = data.promoted_tokens.map(t => `
        <div class="token-row">
          <span>"${t.token}"</span>
          <span class="token-pos">+${t.logit.toFixed(3)}</span>
        </div>
      `).join("");
    }

    const suppList = document.getElementById("suppressed-list");
    if (suppList) {
      suppList.innerHTML = data.suppressed_tokens.map(t => `
        <div class="token-row">
          <span>"${t.token}"</span>
          <span class="token-neg">${t.logit.toFixed(3)}</span>
        </div>
      `).join("");
    }
  } catch (err) {
    console.error("Failed to load logits:", err);
  }
}

function initSteeringControls() {
  const steerBtn = document.getElementById("btn-steer");
  const alphaSlider = document.getElementById("steer-alpha");
  const alphaLabel = document.getElementById("alpha-val");
  const outputBox = document.getElementById("steer-output");
  const outputText = document.getElementById("steered-text");

  if (alphaSlider && alphaLabel) {
    alphaSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      alphaLabel.innerText = `${val >= 0 ? "+" : ""}${val.toFixed(1)}`;
    });
  }

  if (steerBtn) {
    steerBtn.addEventListener("click", async () => {
      const promptInput = document.getElementById("steer-prompt");
      const featInput = document.getElementById("steer-feat-id");
      const prompt = promptInput ? promptInput.value : "";
      const featId = featInput ? parseInt(featInput.value, 10) : 0;
      const alpha = alphaSlider ? parseFloat(alphaSlider.value) : 5.0;

      if (isNaN(featId) || !prompt) return;

      if (outputBox) outputBox.classList.remove("hidden");
      if (outputText) outputText.innerText = "Steering model generation in progress...";

      try {
        const res = await fetch("/api/steer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: prompt,
            steered_features: { [featId]: alpha },
            max_new_tokens: 40,
            temperature: 0.7,
            layer: currentActiveLayer
          })
        });

        if (!res.ok) throw new Error("Steering request failed.");
        const data = await res.json();
        if (outputText) outputText.innerText = data.generated_text;
      } catch (err) {
        if (outputText) outputText.innerText = `Error: ${err.message}`;
      }
    });
  }
}

function initAnalyze() {
  const analyzeBtn = document.getElementById("btn-analyze");
  const analyzeInput = document.getElementById("analyze-input");
  const resultsCard = document.getElementById("analyze-results");
  const container = document.getElementById("heatmaps-container");

  if (analyzeBtn && analyzeInput) {
    analyzeBtn.addEventListener("click", async () => {
      const text = analyzeInput.value;
      if (!text) return;

      if (resultsCard) resultsCard.classList.remove("hidden");
      if (container) container.innerHTML = '<div class="loading">Analyzing activations...</div>';

      try {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text, top_k_features: 5, layer: currentActiveLayer })
        });

        if (!res.ok) throw new Error("Analysis failed.");
        const data = await res.json();
        if (container) {
          container.innerHTML = "";

          data.top_features.forEach(feat => {
            const card = document.createElement("div");
            card.className = "card";
            card.style.marginTop = "12px";
            
            const maxAct = Math.max(...feat.token_activations.map(t => t.act), 1e-6);
            const tokenSpans = feat.token_activations.map(t => {
              const intensity = Math.min(1, Math.max(0, t.act / maxAct));
              const bg = `rgba(59, 130, 246, ${intensity * 0.8})`;
              return `<span class="token-pill" style="background:${bg}; padding: 3px 6px; border-radius: 4px; margin: 2px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; display: inline-block;">${t.token}</span>`;
            }).join("");

            card.innerHTML = `
              <h4 style="margin-bottom:4px; color:#60a5fa;">Feature #${feat.feature_id} — ${feat.explanation}</h4>
              <p style="font-size:0.8rem; color:#a1a1aa; margin-bottom:8px;">Mean Activation: ${feat.mean_activation.toFixed(3)}</p>
              <div style="line-height: 2.2; background:#18181b; padding:10px; border-radius:6px; border: 1px solid #27272a;">${tokenSpans}</div>
            `;
            container.appendChild(card);
          });
        }
      } catch (err) {
        if (container) container.innerHTML = `<div class="card"><p style="color:#ef4444">Error: ${err.message}</p></div>`;
      }
    });
  }
}
