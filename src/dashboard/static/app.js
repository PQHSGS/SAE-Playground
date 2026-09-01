let currentSelectedFeature = 0;
let currentActiveLayer = "";
let analyzedTokensData = [];

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

  // Highlight in left sidebar
  document.querySelectorAll(".feature-item").forEach(item => {
    if (item.querySelector(".feat-id-tag")?.innerText === `Feature #${featureId}`) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Fetch Full Feature Details (Neuronpedia Context Snippets + Logits)
  try {
    const res = await fetch(`/api/feature/${featureId}?top_k=10&layer=${encodeURIComponent(currentActiveLayer)}`);
    if (!res.ok) return;
    const data = await res.json();

    if (titleEl) {
      titleEl.innerText = (data.title && data.title !== `Feature #${featureId}`) 
        ? `${data.title} (#${featureId})` 
        : `Feature #${featureId}`;
    }
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
};
