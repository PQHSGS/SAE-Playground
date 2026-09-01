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
        const layerBadge = document.getElementById("hero-layer-badge");
        if (layerBadge) layerBadge.innerText = currentActiveLayer;
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
  listEl.innerHTML = '<div class="loading">Loading feature catalog...</div>';
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
        selectFeature(feat.feature_id);
      });
      listEl.appendChild(item);
    });

    if (data.features.length > 0) {
      selectFeature(data.features[0].feature_id);
    }
  } catch (err) {
    listEl.innerHTML = `<div class="feature-item"><p style="color:#ef4444">Could not load features: ${err.message}</p></div>`;
  }
}

async function selectFeature(featureId) {
  currentSelectedFeature = featureId;
  const titleEl = document.getElementById("inspect-title");
  const expEl = document.getElementById("inspect-explanation");
  const layerBadge = document.getElementById("hero-layer-badge");
  const maxActEl = document.getElementById("stat-max-act");
  const firingRateEl = document.getElementById("stat-firing-rate");
  const steerFeatInput = document.getElementById("steer-feat-id");

  if (titleEl) titleEl.innerText = `Feature #${featureId}`;
  if (layerBadge) layerBadge.innerText = currentActiveLayer;
  if (steerFeatInput) steerFeatInput.value = featureId;

  // Fetch Full Feature Details (Neuronpedia Context Snippets + Logits)
  try {
    const res = await fetch(`/api/feature/${featureId}?top_k=10&layer=${encodeURIComponent(currentActiveLayer)}`);
    if (!res.ok) return;
    const data = await res.json();

    if (expEl) expEl.innerText = data.explanation || `Feature #${featureId}`;
    if (maxActEl) maxActEl.innerText = `+${data.max_activation.toFixed(2)}`;
    if (firingRateEl) firingRateEl.innerText = `${(data.firing_rate * 100).toFixed(3)}%`;

    // 1. Render Feature 1: Top Activating Context Snippets (Neuronpedia Core)
    const snippetsContainer = document.getElementById("snippets-container");
    if (snippetsContainer) {
      if (!data.snippets || data.snippets.length === 0) {
        snippetsContainer.innerHTML = '<div class="empty-msg">No activating dataset examples recorded for this feature.</div>';
      } else {
        snippetsContainer.innerHTML = data.snippets.map((snip, sIdx) => {
          const maxSnippetAct = Math.max(...snip.tokens.map(t => t.act), 1e-5);
          const tokenSpans = snip.tokens.map(t => {
            const intensity = Math.min(1.0, Math.max(0.0, t.act / maxSnippetAct));
            const isPeak = t.act >= maxSnippetAct * 0.95 && t.act > 0.1;
            const bg = intensity > 0.05 ? `rgba(139, 92, 246, ${Math.max(0.2, intensity * 0.9)})` : 'transparent';
            const color = intensity > 0.3 ? '#ffffff' : '#cbd5e1';
            const peakClass = isPeak ? 'peak-token' : '';
            return `<span class="token-pill ${peakClass}" style="background:${bg}; color:${color};" title="act: ${t.act.toFixed(3)}">${t.token}</span>`;
          }).join("");

          return `
            <div class="snippet-card">
              <div class="snippet-meta">
                <span style="color:#94a3b8; font-weight:600;">Example #${sIdx + 1}</span>
                <span class="max-act-badge">Max Act: +${snip.max_activation.toFixed(2)}</span>
              </div>
              <div class="snippet-text-box">${tokenSpans}</div>
            </div>
          `;
        }).join("");
      }
    }

    // 2. Render Feature 2: Direct Logit Attribution (Logit Lens)
    const promList = document.getElementById("promoted-list");
    if (promList) {
      const maxProm = data.promoted_tokens.length > 0 ? Math.max(...data.promoted_tokens.map(t => t.logit), 1e-5) : 1;
      promList.innerHTML = data.promoted_tokens.map(t => {
        const widthPct = Math.min(100, Math.max(5, (t.logit / maxProm) * 100));
        return `
          <div class="token-row">
            <div class="logit-bar-pos" style="width: ${widthPct}%;"></div>
            <span class="token-str">"${t.token}"</span>
            <span class="token-pos">+${t.logit.toFixed(3)}</span>
          </div>
        `;
      }).join("");
    }

    const suppList = document.getElementById("suppressed-list");
    if (suppList) {
      const maxSupp = data.suppressed_tokens.length > 0 ? Math.max(...data.suppressed_tokens.map(t => Math.abs(t.logit)), 1e-5) : 1;
      suppList.innerHTML = data.suppressed_tokens.map(t => {
        const widthPct = Math.min(100, Math.max(5, (Math.abs(t.logit) / maxSupp) * 100));
        return `
          <div class="token-row">
            <div class="logit-bar-neg" style="width: ${widthPct}%;"></div>
            <span class="token-str">"${t.token}"</span>
            <span class="token-neg">${t.logit.toFixed(3)}</span>
          </div>
        `;
      }).join("");
    }
  } catch (err) {
    console.error("Failed to load feature details:", err);
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
            max_new_tokens: 35,
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
              const intensity = Math.min(1.0, Math.max(0.0, t.act / maxAct));
              const bg = intensity > 0.05 ? `rgba(139, 92, 246, ${Math.max(0.2, intensity * 0.85)})` : 'transparent';
              const color = intensity > 0.3 ? '#ffffff' : '#cbd5e1';
              return `<span class="token-pill" style="background:${bg}; color:${color}; font-family: 'JetBrains Mono', monospace; font-size: 13px; display: inline-block; margin: 1px;" title="act: ${t.act.toFixed(3)}">${t.token}</span>`;
            }).join("");

            card.innerHTML = `
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="color:#a78bfa; font-family:'JetBrains Mono', monospace; font-size:14px;">Feature #${feat.feature_id}</h4>
                <span class="max-act-badge">Mean Act: +${feat.mean_activation.toFixed(2)}</span>
              </div>
              <p style="font-size:12px; color:#94a3b8; margin-bottom:10px;">${feat.explanation}</p>
              <div class="snippet-text-box">${tokenSpans}</div>
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
