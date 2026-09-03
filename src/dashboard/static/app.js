let currentSelectedFeature = 0;
let currentActiveLayer = "";
let analyzedTokensData = [];
const featureDetailsCache = new Map();

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initLayers();
  initSearch();
  initSteeringControls();
  initAnalyze();
  initAttributionGraph();
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
  const modelTag = document.getElementById("current-model-tag");
  try {
    try {
      const healthRes = await fetch("/api/health");
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        if (modelTag && healthData.model_name) {
          modelTag.innerText = healthData.model_name.replace("cattonpm/", "").replace("google/", "");
        }
      }
    } catch (e) {
      console.warn("Could not fetch health:", e);
    }

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
    }, 200);
  });
}

async function loadFeatures(searchQuery = "") {
  const listEl = document.getElementById("feature-list");
  listEl.innerHTML = '<div class="loading">Loading feature catalog...</div>';
  try {
    let url = `/api/features?page=1&page_size=50&layer=${encodeURIComponent(currentActiveLayer)}`;
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

    const fragment = document.createDocumentFragment();
    data.features.forEach((feat, idx) => {
      const item = document.createElement("div");
      item.className = `feature-item ${idx === 0 ? "active" : ""}`;
      const titleText = feat.title || feat.explanation || `Feature #${feat.feature_id}`;
      item.innerHTML = `
        <div class="feature-item-top-label">
          <span class="label-pill" title="${escapeHtml(feat.explanation || titleText)}">${escapeHtml(titleText)}</span>
        </div>
        <div class="feature-item-sub">
          <span class="feat-id-tag">Feature #${feat.feature_id}</span>
          ${feat.max_activation ? `<span class="max-act-badge">+${feat.max_activation.toFixed(2)}</span>` : ""}
        </div>
      `;
      item.addEventListener("click", () => {
        document.querySelectorAll(".feature-item").forEach(i => i.classList.remove("active"));
        item.classList.add("active");
        selectFeature(feat.feature_id);
      });
      fragment.appendChild(item);
    });
    listEl.appendChild(fragment);

    if (data.features.length > 0) {
      selectFeature(data.features[0].feature_id);
    }
  } catch (err) {
    listEl.innerHTML = `<div class="feature-item"><p style="color:#ef4444">Could not load features: ${err.message}</p></div>`;
  }
}

async function selectFeature(featureId) {
  currentSelectedFeature = featureId;

  // Highlight in left sidebar
  document.querySelectorAll(".feature-item").forEach(item => {
    if (item.querySelector(".feat-id-tag")?.innerText === `Feature #${featureId}`) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  const cacheKey = `${currentActiveLayer}_${featureId}`;
  let data = featureDetailsCache.get(cacheKey);

  if (!data) {
    try {
      const res = await fetch(`/api/feature/${featureId}?top_k=10&layer=${encodeURIComponent(currentActiveLayer)}`);
      if (!res.ok) return;
      data = await res.json();
      featureDetailsCache.set(cacheKey, data);
    } catch (err) {
      console.error("Failed to load feature details:", err);
      return;
    }
  }

  // Unified Tab 1 Renderers
  renderHeroCard(data, featureId);
  renderContextSnippets(data.snippets);
  renderLogitAttributionList("promoted-list", data.promoted_tokens, true);
  renderLogitAttributionList("suppressed-list", data.suppressed_tokens, false);
}

function renderHeroCard(data, featureId) {
  const titleEl = document.getElementById("inspect-title");
  const expEl = document.getElementById("inspect-explanation");
  const layerBadge = document.getElementById("hero-layer-badge");
  const maxActEl = document.getElementById("stat-max-act");
  const firingRateEl = document.getElementById("stat-firing-rate");
  const steerFeatInput = document.getElementById("steer-feat-id");

  if (layerBadge) layerBadge.innerText = currentActiveLayer;
  if (steerFeatInput) steerFeatInput.value = featureId;

  const displayTitle = (data.title && data.title !== `Feature #${featureId}`)
    ? `${data.title} (#${featureId})`
    : `Feature #${featureId}`;

  if (titleEl) titleEl.innerText = displayTitle;
  if (expEl) expEl.innerText = data.explanation || `Feature #${featureId}`;
  if (maxActEl) maxActEl.innerText = `+${data.max_activation.toFixed(2)}`;
  if (firingRateEl) firingRateEl.innerText = `${(data.firing_rate * 100).toFixed(3)}%`;
}

function renderContextSnippets(snippets) {
  const container = document.getElementById("snippets-container");
  if (!container) return;

  if (!snippets || snippets.length === 0) {
    container.innerHTML = '<div class="empty-msg">No activating dataset examples recorded for this feature.</div>';
    return;
  }

  container.innerHTML = snippets.map((snip, sIdx) => {
    const maxSnippetAct = Math.max(...snip.tokens.map(t => t.act), 1e-5);
    const tokenSpans = snip.tokens.map(t => {
      const intensity = Math.min(1.0, Math.max(0.0, t.act / maxSnippetAct));
      const isPeak = t.act >= maxSnippetAct * 0.95 && t.act > 0.1;
      const bg = intensity > 0.05 ? `rgba(139, 92, 246, ${Math.max(0.2, intensity * 0.9)})` : 'transparent';
      const color = intensity > 0.3 ? '#ffffff' : '#cbd5e1';
      const peakClass = isPeak ? 'peak-token' : '';
      return `<span class="token-pill ${peakClass}" style="background:${bg}; color:${color};" title="Activation: +${t.act.toFixed(3)}">${escapeHtml(t.token)}</span>`;
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

function renderLogitAttributionList(containerId, tokens, isPositive) {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!tokens || tokens.length === 0) {
    container.innerHTML = '<div class="empty-msg">No logit attributions recorded.</div>';
    return;
  }

  const maxVal = Math.max(...tokens.map(t => Math.abs(t.logit)), 1e-5);
  const barClass = isPositive ? "logit-bar-pos" : "logit-bar-neg";
  const valClass = isPositive ? "token-pos" : "token-neg";
  const sign = isPositive ? "+" : "";

  container.innerHTML = tokens.map(t => {
    const widthPct = Math.min(100, Math.max(5, (Math.abs(t.logit) / maxVal) * 100));
    return `
      <div class="token-row">
        <div class="${barClass}" style="width: ${widthPct}%;"></div>
        <span class="token-str">"${escapeHtml(t.token)}"</span>
        <span class="${valClass}">${sign}${t.logit.toFixed(3)}</span>
      </div>
    `;
  }).join("");
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
      if (container) container.innerHTML = '<div class="loading">Computing token activations...</div>';

      try {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text, top_k_per_token: 10, layer: currentActiveLayer })
        });

        if (!res.ok) throw new Error("Analysis failed.");
        const data = await res.json();
        analyzedTokensData = data.tokens || [];

        if (container) {
          container.innerHTML = "";

          // 1. Render Interactive Clickable Token Chips Bar (Neuronpedia Signature)
          const chipsBox = document.createElement("div");
          chipsBox.className = "token-chips-bar";
          chipsBox.style.background = "#070a12";
          chipsBox.style.padding = "14px";
          chipsBox.style.borderRadius = "8px";
          chipsBox.style.border = "1px solid #1e293b";
          chipsBox.style.lineHeight = "2.4";
          chipsBox.style.marginBottom = "18px";

          const maxSeqAct = Math.max(...analyzedTokensData.map(t => t.max_activation), 1e-4);

          analyzedTokensData.forEach((tok, tIdx) => {
            const chip = document.createElement("span");
            const intensity = Math.min(1.0, Math.max(0.0, tok.max_activation / maxSeqAct));
            const bg = intensity > 0.05 ? `rgba(139, 92, 246, ${Math.max(0.2, intensity * 0.85)})` : '#131b2e';
            chip.className = `token-chip ${tIdx === analyzedTokensData.length - 1 ? "active" : ""}`;
            chip.setAttribute("data-token-idx", tIdx);
            chip.style.padding = "4px 8px";
            chip.style.margin = "2px 3px";
            chip.style.borderRadius = "4px";
            chip.style.fontFamily = "'JetBrains Mono', monospace";
            chip.style.fontSize = "13px";
            chip.style.background = bg;
            chip.style.color = intensity > 0.3 ? "#ffffff" : "#cbd5e1";
            chip.style.cursor = "pointer";
            chip.style.border = tIdx === analyzedTokensData.length - 1 ? "1px solid #a78bfa" : "1px solid transparent";
            chip.style.display = "inline-block";
            chip.innerText = tok.token_str;

            chip.addEventListener("click", () => {
              document.querySelectorAll(".token-chip").forEach(c => {
                c.classList.remove("active");
                c.style.border = "1px solid transparent";
              });
              chip.classList.add("active");
              chip.style.border = "1px solid #a78bfa";
              renderTokenActiveFeatures(tIdx);
            });

            chipsBox.appendChild(chip);
          });

          container.appendChild(chipsBox);

          // 2. Active Features Panel for Selected Token
          const tokenPanel = document.createElement("div");
          tokenPanel.id = "token-active-features-panel";
          container.appendChild(tokenPanel);

          // Default to last token (or highest activating token)
          const defaultIdx = analyzedTokensData.length > 0 ? analyzedTokensData.length - 1 : 0;
          renderTokenActiveFeatures(defaultIdx);
        }
      } catch (err) {
        if (container) container.innerHTML = `<div class="card"><p style="color:#ef4444">Error: ${err.message}</p></div>`;
      }
    });
  }
}

function renderTokenActiveFeatures(tokenIdx) {
  const panel = document.getElementById("token-active-features-panel");
  if (!panel || !analyzedTokensData[tokenIdx]) return;

  const tok = analyzedTokensData[tokenIdx];
  const maxTokenAct = tok.max_activation > 0 ? tok.max_activation : 1;

  let featuresHtml = "";
  if (tok.top_features.length === 0) {
    featuresHtml = '<div class="empty-msg">No active features detected on this token position.</div>';
  } else {
    featuresHtml = tok.top_features.map(f => {
      const barWidth = Math.min(100, Math.max(5, (f.activation / maxTokenAct) * 100));
      const titleText = f.title || `Feature #${f.feature_id}`;
      const descText = f.description || (f.explanation && f.explanation !== titleText ? f.explanation : "");
      const promotedBadges = (f.top_promoted_tokens && f.top_promoted_tokens.length > 0)
        ? `<div style="margin-top:7px; font-size:11.5px; color:#94a3b8; display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
             <span style="font-weight:600; color:#64748b;">Promotes:</span>
             ${f.top_promoted_tokens.slice(0, 3).map(p => `<span style="background:rgba(16, 185, 129, 0.15); color:#34d399; border:1px solid rgba(16, 185, 129, 0.3); padding:1px 6px; border-radius:3px; font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:600;">"${p.token}" (+${p.logit.toFixed(2)})</span>`).join("")}
           </div>`
        : "";

      return `
        <div class="snippet-card" style="margin-bottom:10px; border-left: 3px solid #8b5cf6; padding:12px 14px;">
          <div style="margin-bottom:8px; display:flex; align-items:center; justify-content:space-between; gap:8px;">
            <span class="label-pill-bright" style="white-space:normal; font-size:13px; font-weight:700; line-height:1.4;" title="${escapeHtml(descText || titleText)}">🏷️ ${escapeHtml(titleText)}</span>
          </div>
          ${descText ? `<p style="font-size:12px; color:#cbd5e1; margin-bottom:8px; line-height:1.4;">${escapeHtml(descText)}</p>` : ""}
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-family:'JetBrains Mono', monospace; font-size:12.5px; font-weight:700; color:#93c5fd;">Feature #${f.feature_id}</span>
              <button onclick="jumpToFeature(${f.feature_id})" style="background:#1e293b; color:#a78bfa; border:1px solid #334155; border-radius:4px; font-size:11px; padding:2px 7px; cursor:pointer; font-weight:600;">Inspect Full Details ↗</button>
            </div>
            <span class="max-act-badge">Act: +${f.activation.toFixed(3)}</span>
          </div>
          <div style="height:4px; background:#070a12; border-radius:2px; overflow:hidden; margin-bottom:4px;">
            <div style="height:100%; width:${barWidth}%; background:linear-gradient(90deg, #8b5cf6, #ec4899); border-radius:2px;"></div>
          </div>
          ${promotedBadges}
        </div>
      `;
    }).join("");
  }

  panel.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #1e293b; padding-bottom:8px;">
      <div>
        <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:700; letter-spacing:0.05em;">Token Focus:</span>
        <h3 style="font-family:'JetBrains Mono', monospace; color:#f8fafc; font-size:16px; margin-top:2px;">"${tok.token_str}" <span style="font-size:12px; color:#64748b; font-weight:400;">(Token #${tok.token_index}, ID: ${tok.token_id})</span></h3>
      </div>
      <span class="count-badge" style="background:#1e293b; color:#a78bfa; font-family:'JetBrains Mono', monospace; font-size:12px; padding:4px 10px;">${tok.total_active_features} Features Fired</span>
    </div>
    <div style="display:flex; flex-direction:column; gap:6px;">
      ${featuresHtml}
    </div>
  `;
}

window.jumpToFeature = function(featureId) {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(t => {
    if (t.getAttribute("data-tab") === "tab-inspect") {
      t.click();
    }
  });
  selectFeature(featureId);
  const heroCard = document.querySelector(".hero-card");
  if (heroCard) {
    heroCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
};

/* =========================================================================
   Tab 4: Anthropic Attribution Graph (Circuit Tracing)
   ========================================================================= */
function initAttributionGraph() {
  const promptInput = document.getElementById("graph-prompt");
  const tauSlider = document.getElementById("graph-tau");
  const tauVal = document.getElementById("graph-tau-val");
  const maxNodesInput = document.getElementById("graph-max-nodes");
  const maxEdgesInput = document.getElementById("graph-max-edges");
  const traceBtn = document.getElementById("btn-trace-graph");
  const loadingSpan = document.getElementById("graph-loading");
  const candidatesBox = document.getElementById("graph-candidates-box");
  const candidatesList = document.getElementById("graph-candidates-list");
  const metricsCard = document.getElementById("graph-metrics-card");
  const canvasContainer = document.getElementById("graph-canvas-container");
  const svg = document.getElementById("attribution-graph-svg");
  const inspector = document.getElementById("graph-node-inspector");
  const inspectorClose = document.getElementById("gnode-close");

  let selectedTargetId = null;

  if (tauSlider && tauVal) {
    tauSlider.addEventListener("input", (e) => {
      tauVal.innerText = parseFloat(e.target.value).toFixed(2);
    });
  }

  if (inspectorClose && inspector) {
    inspectorClose.addEventListener("click", () => {
      inspector.classList.add("hidden");
    });
  }

  async function runTrace() {
    const prompt = promptInput ? promptInput.value.trim() : "";
    if (!prompt) return;

    if (loadingSpan) loadingSpan.classList.remove("hidden");
    if (traceBtn) traceBtn.disabled = true;

    try {
      const res = await fetch("/api/attribution_graph", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          target_token_id: selectedTargetId,
          pruning_threshold: tauSlider ? parseFloat(tauSlider.value) : 0.80,
          max_nodes: maxNodesInput ? parseInt(maxNodesInput.value) : 35,
          max_edges: maxEdgesInput ? parseInt(maxEdgesInput.value) : 45,
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Failed to trace attribution graph.");
      }

      const data = await res.json();
      renderAttributionGraph(data);
    } catch (err) {
      alert("Attribution Graph Error: " + err.message);
    } finally {
      if (loadingSpan) loadingSpan.classList.add("hidden");
      if (traceBtn) traceBtn.disabled = false;
    }
  }

  if (traceBtn) {
    traceBtn.addEventListener("click", () => {
      selectedTargetId = null;
      runTrace();
    });
  }

  function renderAttributionGraph(data) {
    // 1. Populate Metrics
    const targetEl = document.getElementById("graph-metric-target");
    const probEl = document.getElementById("graph-metric-prob");
    const sizeEl = document.getElementById("graph-metric-size");
    const compEl = document.getElementById("graph-metric-comp");

    if (targetEl) targetEl.innerText = `"${data.target_token.trim()}"`;
    if (probEl) probEl.innerText = `${(data.target_prob * 100).toFixed(1)}%`;
    if (sizeEl) sizeEl.innerText = `${data.nodes.length} Nodes, ${data.edges.length} Edges`;
    if (compEl && data.metrics) {
      compEl.innerText = `${(data.metrics.completeness_score * 100).toFixed(1)}%`;
    }

    if (metricsCard) metricsCard.classList.remove("hidden");
    if (canvasContainer) canvasContainer.classList.remove("hidden");

    // 2. Candidate Prediction Pills
    if (candidatesBox && candidatesList && data.candidates && data.candidates.length > 0) {
      candidatesBox.classList.remove("hidden");
      candidatesList.innerHTML = "";
      data.candidates.forEach(cand => {
        const pill = document.createElement("button");
        const isSelected = cand.id === data.target_token_id;
        pill.className = "tab-btn";
        pill.style.padding = "4px 10px";
        pill.style.fontSize = "12px";
        pill.style.fontFamily = "'JetBrains Mono', monospace";
        pill.style.background = isSelected ? "var(--accent)" : "rgba(255,255,255,0.06)";
        pill.style.color = isSelected ? "#fff" : "var(--text)";
        pill.innerText = `"${cand.token.trim()}" (${(cand.prob * 100).toFixed(1)}%)`;

        pill.addEventListener("click", () => {
          selectedTargetId = cand.id;
          runTrace();
        });
        candidatesList.appendChild(pill);
      });
    }

    // 3. Layout DAG in SVG
    if (!svg) return;
    svg.innerHTML = "";

    // Group nodes by layer_idx
    const layerGroups = {};
    data.nodes.forEach(n => {
      const l = n.layer_idx;
      if (!layerGroups[l]) layerGroups[l] = [];
      layerGroups[l].push(n);
    });

    const sortedLayers = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);
    const colWidth = 220;
    const nodeWidth = 170;
    const nodeHeight = 54;
    const vertGap = 16;
    const startX = 40;
    const startY = 40;

    let maxNodesInCol = 0;
    sortedLayers.forEach(l => {
      if (layerGroups[l].length > maxNodesInCol) {
        maxNodesInCol = layerGroups[l].length;
      }
    });

    const totalWidth = Math.max(920, startX * 2 + (sortedLayers.length - 1) * colWidth + nodeWidth);
    const totalHeight = Math.max(540, startY * 2 + maxNodesInCol * (nodeHeight + vertGap));

    svg.setAttribute("viewBox", `0 0 ${totalWidth} ${totalHeight}`);
    svg.style.width = `${totalWidth}px`;
    svg.style.height = `${totalHeight}px`;

    // Calculate node coordinates (x, y)
    const nodeCoords = {};
    sortedLayers.forEach((l, colIdx) => {
      const nodesInCol = layerGroups[l];
      const colX = startX + colIdx * colWidth;
      const colHeight = nodesInCol.length * (nodeHeight + vertGap) - vertGap;
      const colStartY = Math.max(startY, (totalHeight - colHeight) / 2);

      nodesInCol.forEach((n, rowIdx) => {
        const y = colStartY + rowIdx * (nodeHeight + vertGap);
        nodeCoords[n.id] = {
          x: colX,
          y: y,
          w: nodeWidth,
          h: nodeHeight,
          node: n,
        };
      });
    });

    // 4. Render Column Headers
    sortedLayers.forEach((l, colIdx) => {
      const colX = startX + colIdx * colWidth;
      const header = document.createElementNS("http://www.w3.org/2000/svg", "text");
      header.setAttribute("x", colX + nodeWidth / 2);
      header.setAttribute("y", 24);
      header.setAttribute("text-anchor", "middle");
      header.setAttribute("fill", "#64748b");
      header.setAttribute("font-size", "11px");
      header.setAttribute("font-weight", "700");
      header.setAttribute("letter-spacing", "0.05em");
      header.textContent = l === -1 ? "INPUT TOKENS" : (l === sortedLayers[sortedLayers.length - 1] ? "TARGET LOGIT" : `LAYER ${l}`);
      svg.appendChild(header);
    });

    // 5. Render Edges (Curved Bezier Paths)
    const maxWeight = Math.max(...data.edges.map(e => e.weight), 1e-4);
    const edgeElements = [];

    data.edges.forEach(e => {
      const s = nodeCoords[e.source];
      const t = nodeCoords[e.target];
      if (!s || !t) return;

      const sx = s.x + s.w;
      const sy = s.y + s.h / 2;
      const tx = t.x;
      const ty = t.y + t.h / 2;

      const normW = Math.max(0.1, Math.min(1.0, e.weight / maxWeight));
      const strokeW = Math.max(1.5, normW * 5.5);
      const alpha = Math.max(0.2, normW * 0.85);

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const c1x = sx + (tx - sx) * 0.5;
      const c1y = sy;
      const c2x = sx + (tx - sx) * 0.5;
      const c2y = ty;

      path.setAttribute("d", `M ${sx} ${sy} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${tx} ${ty}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", `rgba(139, 92, 246, ${alpha})`);
      path.setAttribute("stroke-width", strokeW);
      path.setAttribute("data-source", e.source);
      path.setAttribute("data-target", e.target);
      path.classList.add("circuit-edge");

      svg.appendChild(path);
      edgeElements.push({ path, source: e.source, target: e.target });
    });

    // 6. Render Nodes
    data.nodes.forEach(n => {
      const c = nodeCoords[n.id];
      if (!c) return;

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "circuit-node");
      g.setAttribute("data-id", n.id);
      g.style.cursor = "pointer";

      // Pill Background
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", c.x);
      rect.setAttribute("y", c.y);
      rect.setAttribute("width", c.w);
      rect.setAttribute("height", c.h);
      rect.setAttribute("rx", "6");
      rect.setAttribute("ry", "6");

      let strokeCol = "#334155";
      let fillCol = "#0f172a";
      if (n.node_type === "input_token") {
        strokeCol = "#3b82f6";
        fillCol = "#172554";
      } else if (n.node_type === "logit") {
        strokeCol = "#10b981";
        fillCol = "#064e3b";
      } else {
        strokeCol = "#8b5cf6";
        fillCol = "#1e1b4b";
      }

      rect.setAttribute("fill", fillCol);
      rect.setAttribute("stroke", strokeCol);
      rect.setAttribute("stroke-width", "1.5");
      g.appendChild(rect);

      // Node Label (Token or Feature ID)
      const textLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      textLabel.setAttribute("x", c.x + 8);
      textLabel.setAttribute("y", c.y + 18);
      textLabel.setAttribute("fill", "#f8fafc");
      textLabel.setAttribute("font-size", "12px");
      textLabel.setAttribute("font-family", "'JetBrains Mono', monospace");
      textLabel.setAttribute("font-weight", "700");
      textLabel.textContent = n.label.length > 18 ? n.label.substring(0, 16) + "…" : n.label;
      g.appendChild(textLabel);

      // Node Title (Conceptual Semantic Tag)
      const textTitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
      textTitle.setAttribute("x", c.x + 8);
      textTitle.setAttribute("y", c.y + 34);
      textTitle.setAttribute("fill", "#94a3b8");
      textTitle.setAttribute("font-size", "10px");
      textTitle.textContent = n.title.length > 22 ? n.title.substring(0, 20) + "…" : n.title;
      g.appendChild(textTitle);

      // Node Influence / Activation Value
      const textVal = document.createElementNS("http://www.w3.org/2000/svg", "text");
      textVal.setAttribute("x", c.x + 8);
      textVal.setAttribute("y", c.y + 47);
      textVal.setAttribute("fill", "#a78bfa");
      textVal.setAttribute("font-size", "9px");
      textVal.setAttribute("font-family", "'JetBrains Mono', monospace");
      textVal.textContent = n.node_type === "feature"
        ? `act: ${n.activation.toFixed(2)} | inf: ${n.logit_influence.toFixed(3)}`
        : (n.node_type === "input_token" ? `pos: ${n.pos}` : `prob: ${(n.activation * 100).toFixed(1)}%`);
      g.appendChild(textVal);

      // Interaction: Click to inspect and highlight connected pathways
      g.addEventListener("click", () => {
        // Highlight connected edges
        edgeElements.forEach(({ path, source, target }) => {
          if (source === n.id) {
            path.setAttribute("stroke", "#38bdf8"); // Cyan outgoing
            path.setAttribute("stroke-width", "4");
          } else if (target === n.id) {
            path.setAttribute("stroke", "#f59e0b"); // Orange incoming
            path.setAttribute("stroke-width", "4");
          } else {
            path.setAttribute("stroke", "rgba(71, 85, 105, 0.2)");
            path.setAttribute("stroke-width", "1");
          }
        });

        // Open Inspector Drawer
        if (inspector) {
          inspector.classList.remove("hidden");
          const tagEl = document.getElementById("gnode-tag");
          const titleEl = document.getElementById("gnode-title");
          const descEl = document.getElementById("gnode-desc");
          const actEl = document.getElementById("gnode-act");
          const infEl = document.getElementById("gnode-inf");
          const promEl = document.getElementById("gnode-promoted");

          if (tagEl) tagEl.innerText = n.node_type.toUpperCase() + (n.layer ? ` (${n.layer})` : "");
          if (titleEl) titleEl.innerText = n.title;
          if (descEl) descEl.innerText = n.explanation || "No description available.";
          if (actEl) actEl.innerText = n.activation.toFixed(4);
          if (infEl) infEl.innerText = n.logit_influence.toFixed(4);
          if (promEl) {
            promEl.innerHTML = n.promoted_tokens && n.promoted_tokens.length > 0
              ? n.promoted_tokens.map(t => `<span class="tag" style="background:#1e293b; color:#38bdf8; margin-right:4px;">"${typeof t === 'object' ? t.token : t}"</span>`).join("")
              : '<span style="color:#64748b;">N/A</span>';
          }
          inspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      });

      svg.appendChild(g);
    });
  }
}

