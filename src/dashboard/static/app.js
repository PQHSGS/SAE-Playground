let currentSelectedFeature = 0;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadFeatures();
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
      document.getElementById(targetId).classList.add("active");
    });
  });
}

async function loadFeatures() {
  const listEl = document.getElementById("feature-list");
  try {
    const res = await fetch("/api/features?page=1&page_size=100");
    if (!res.ok) throw new Error("Backend not initialized or no model loaded.");
    const data = await res.json();
    document.getElementById("total-count").innerText = `${data.total_features} Features`;
    listEl.innerHTML = "";

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
  document.getElementById("inspect-title").innerText = `Feature #${featureId}`;
  document.getElementById("inspect-explanation").innerText = explanation || `Feature #${featureId}`;
  document.getElementById("steer-feat-id").value = featureId;

  // Fetch Logit Lens Attribution
  try {
    const res = await fetch(`/api/feature/${featureId}/logits?top_k=10`);
    if (!res.ok) return;
    const data = await res.json();

    const promList = document.getElementById("promoted-list");
    promList.innerHTML = data.promoted_tokens.map(t => `
      <div class="token-row">
        <span>"${t.token}"</span>
        <span class="token-pos">+${t.logit.toFixed(3)}</span>
      </div>
    `).join("");

    const suppList = document.getElementById("suppressed-list");
    suppList.innerHTML = data.suppressed_tokens.map(t => `
      <div class="token-row">
        <span>"${t.token}"</span>
        <span class="token-neg">${t.logit.toFixed(3)}</span>
      </div>
    `).join("");
  } catch (err) {
    console.error("Logit lens fetch failed:", err);
  }
}

function initSteeringControls() {
  const alphaSlider = document.getElementById("steer-alpha");
  const alphaLabel = document.getElementById("alpha-val");
  alphaSlider.addEventListener("input", () => {
    const val = parseFloat(alphaSlider.value);
    alphaLabel.innerText = (val >= 0 ? "+" : "") + val.toFixed(1);
  });

  document.getElementById("btn-steer").addEventListener("click", async () => {
    const prompt = document.getElementById("steer-prompt").value;
    const featId = parseInt(document.getElementById("steer-feat-id").value);
    const alpha = parseFloat(alphaSlider.value);

    const btn = document.getElementById("btn-steer");
    btn.innerText = "Generating...";
    btn.disabled = true;

    try {
      const res = await fetch("/api/steer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          steered_features: { [featId]: alpha },
          max_new_tokens: 35
        })
      });
      const data = await res.json();
      document.getElementById("steer-output").classList.remove("hidden");
      document.getElementById("steered-text").innerText = data.generated_text;
    } catch (err) {
      alert("Steering generation error: " + err.message);
    } finally {
      btn.innerText = "Generate with Steering";
      btn.disabled = false;
    }
  });
}

function initAnalyze() {
  document.getElementById("btn-analyze").addEventListener("click", async () => {
    const text = document.getElementById("analyze-input").value;
    const btn = document.getElementById("btn-analyze");
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    try {
      const res = await fetch("/api/analyze_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, top_k_features: 4 })
      });
      const data = await res.json();
      const container = document.getElementById("heatmaps-container");
      container.innerHTML = "";

      data.top_features.forEach(feat => {
        const box = document.createElement("div");
        box.className = "heatmap-box";
        const tokenSpans = feat.tokens.map(t => {
          const intensity = Math.min(1, Math.max(0, t.activation / 4.0));
          const bg = intensity > 0.05 ? `rgba(99, 102, 241, ${intensity})` : "transparent";
          return `<span class="token-span" style="background:${bg}; border-bottom: 1px solid rgba(255,255,255,${intensity});" title="Act: ${t.activation.toFixed(3)}">${escapeHtml(t.token)}</span>`;
        }).join("");

        box.innerHTML = `
          <h4>Feature #${feat.feature_id}: <span style="font-weight:normal; color:#94a3b8;">${feat.explanation}</span></h4>
          <div style="margin-top:8px;">${tokenSpans}</div>
        `;
        container.appendChild(box);
      });

      document.getElementById("analyze-results").classList.remove("hidden");
    } catch (err) {
      alert("Analysis failed: " + err.message);
    } finally {
      btn.innerText = "Analyze Activations";
      btn.disabled = false;
    }
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
