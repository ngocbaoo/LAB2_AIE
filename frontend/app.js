const COLORS = {
  DQN: "#3b82f6",
  REINFORCE: "#f5720e",
  "Actor-Critic": "#16a34a",
};
const LABELS_ORDER = ["DQN", "REINFORCE", "Actor-Critic"];
const BASE_DELAY_MS = 20;

let DATA = null;

async function loadData() {
  try {
    const res = await fetch("./data/results.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    DATA = await res.json();
  } catch (err) {
    document.querySelectorAll(".status-msg").forEach((el) => {
      el.textContent =
        "Không tải được data/results.json — hãy chạy backend/train_all.py trước, rồi mở trang này qua một local server " +
        "(vd: `py -3.11 -m http.server` trong thư mục frontend) thay vì mở file trực tiếp.";
    });
    console.error(err);
    return;
  }
  initTabs();
  buildPlaybackView(document.getElementById("baseline-tab"), "baseline");
  buildPlaybackView(document.getElementById("improved-tab"), "improved");
  buildCompareTab();
}

// ---------- tabs ----------
function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

// ---------- shared drawing helpers ----------
function drawAxes(ctx, padding, w, h, maxY) {
  ctx.strokeStyle = "#8892a6";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, h - padding.bottom);
  ctx.lineTo(w - padding.right, h - padding.bottom);
  ctx.stroke();

  ctx.fillStyle = "#8892a6";
  ctx.font = "12px sans-serif";
  const ySteps = 5;
  for (let i = 0; i <= ySteps; i++) {
    const val = Math.round((maxY / ySteps) * i);
    const y = h - padding.bottom - ((h - padding.top - padding.bottom) * i) / ySteps;
    ctx.fillText(val, 6, y + 4);
    ctx.strokeStyle = "#8892a61a";
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }
}

function toXY(i, v, w, h, padding, maxX, maxY) {
  const x = padding.left + ((w - padding.left - padding.right) * i) / (maxX - 1 || 1);
  const y = h - padding.bottom - ((h - padding.top - padding.bottom) * v) / maxY;
  return [x, y];
}

function plotLine(ctx, points, color, w, h, padding, maxX, maxY, { alpha = 1, dashed = false, dot = true } = {}) {
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = alpha >= 1 ? 2 : 1;
  if (dashed) ctx.setLineDash([6, 4]);
  ctx.beginPath();
  points.forEach((v, i) => {
    const [x, y] = toXY(i, v, w, h, padding, maxX, maxY);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  if (dot) {
    const lastIdx = points.length - 1;
    const [x, y] = toXY(lastIdx, points[lastIdx], w, h, padding, maxX, maxY);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, alpha >= 1 ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function plotBand(ctx, meanArr, stdArr, color, w, h, padding, maxX, maxY, alpha) {
  if (!meanArr.length) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  meanArr.forEach((v, i) => {
    const [x, y] = toXY(i, v + stdArr[i], w, h, padding, maxX, maxY);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  for (let i = meanArr.length - 1; i >= 0; i--) {
    const [x, y] = toXY(i, meanArr[i] - stdArr[i], w, h, padding, maxX, maxY);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawThreshold(ctx, w, h, padding, maxY, threshold) {
  const [, y] = toXY(0, threshold, w, h, padding, 2, maxY);
  ctx.save();
  ctx.strokeStyle = "#8892a6";
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(padding.left, y);
  ctx.lineTo(w - padding.right, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#8892a6";
  ctx.font = "12px sans-serif";
  ctx.fillText(`ngưỡng hội tụ (${threshold})`, padding.left + 8, y - 6);
  ctx.restore();
}

function keyOf(run) {
  return run.key;
}

function fmtMeanStd(obj, decimals = 1) {
  if (!obj || obj.mean === null || obj.mean === undefined) return "chưa hội tụ";
  if (obj.std === null || obj.n <= 1) return obj.mean.toFixed(decimals);
  return `${obj.mean.toFixed(decimals)} ± ${obj.std.toFixed(decimals)}`;
}

// ---------- baseline / improved playback view ----------
function buildPlaybackView(container, variant) {
  const runs = DATA.runs.filter((r) => r.variant === variant);
  const maxFrame = Math.max(...runs.map((r) => r.moving_avg_mean.length)) - 1;

  const state = {
    visible: new Set(LABELS_ORDER),
    showRaw: false,
    frame: maxFrame,
    maxFrame,
    playing: false,
    timerId: null,
  };

  const canvas = container.querySelector(".reward-chart");
  const ctx = canvas.getContext("2d");
  const statusEl = container.querySelector(".status-msg");
  const playBtn = container.querySelector(".play-btn");
  const resetBtn = container.querySelector(".reset-btn");
  const slider = container.querySelector(".episode-slider");
  const episodeLabel = container.querySelector(".episode-label");
  const speedSelect = container.querySelector(".speed-select");
  const liveStatsEl = container.querySelector(".live-stats");
  const rawToggle = container.querySelector(".show-raw");

  statusEl.textContent = `Môi trường ${DATA.environment} · ${DATA.n_episodes} episodes · ${DATA.seeds.length} seed (${DATA.seeds.join(", ")})`;
  slider.max = String(maxFrame);
  slider.value = String(state.frame);

  liveStatsEl.innerHTML = "";
  runs.forEach((run) => {
    const key = keyOf(run);
    const card = document.createElement("div");
    card.className = "live-card";
    card.style.setProperty("--stat-color", COLORS[key]);
    card.dataset.series = key;
    card.innerHTML = `
      <div class="live-name"><span>${run.label}</span><span class="badge-solved" hidden>hội tụ</span></div>
      <div class="live-reward">—</div>
      <div class="live-sub">reward moving-avg (mean qua seed)</div>
    `;
    liveStatsEl.appendChild(card);
  });

  function render() {
    const w = canvas.width;
    const h = canvas.height;
    const padding = { left: 40, right: 16, top: 16, bottom: 24 };
    const upto = state.frame + 1;

    ctx.clearRect(0, 0, w, h);

    const visibleRuns = runs.filter((r) => state.visible.has(keyOf(r)));
    const maxX = state.maxFrame + 1;
    const maxY = Math.max(...runs.flatMap((r) => r.moving_avg_mean.map((v, i) => v + r.moving_avg_std[i])), DATA.solved_threshold, 10) * 1.05;

    drawAxes(ctx, padding, w, h, maxY);
    drawThreshold(ctx, w, h, padding, maxY, DATA.solved_threshold);

    visibleRuns.forEach((run) => {
      const color = COLORS[keyOf(run)];
      const meanSlice = run.moving_avg_mean.slice(0, upto);
      const stdSlice = run.moving_avg_std.slice(0, upto);
      plotBand(ctx, meanSlice, stdSlice, color, w, h, padding, maxX, maxY, 0.14);
      if (state.showRaw) {
        plotLine(ctx, run.rewards_mean.slice(0, upto), color, w, h, padding, maxX, maxY, { alpha: 0.3, dot: false });
      }
      plotLine(ctx, meanSlice, color, w, h, padding, maxX, maxY, { alpha: 1 });
    });

    episodeLabel.textContent = `episode ${state.frame} / ${state.maxFrame}`;

    runs.forEach((run) => {
      const key = keyOf(run);
      const card = liveStatsEl.querySelector(`[data-series="${key}"]`);
      if (!card) return;
      const idx = Math.min(state.frame, run.moving_avg_mean.length - 1);
      card.querySelector(".live-reward").textContent = run.moving_avg_mean[idx].toFixed(1);
      const solvedMean = run.metrics.solved_at_episode.mean;
      const solvedNow = solvedMean !== null && idx >= solvedMean;
      card.querySelector(".badge-solved").hidden = !solvedNow;
      card.style.opacity = state.visible.has(key) ? "1" : "0.4";
    });
  }

  function renderTable() {
    const tbody = container.querySelector(".metrics-table tbody");
    tbody.innerHTML = "";
    runs.forEach((run) => {
      const m = run.metrics;
      const key = keyOf(run);
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span style="color:${COLORS[key]}; font-weight:600;">●</span> ${run.label}</td>
        <td>${Math.round(m.solved_rate * DATA.seeds.length)}/${DATA.seeds.length}</td>
        <td>${fmtMeanStd(m.solved_at_episode, 0)}</td>
        <td>${fmtMeanStd(m.env_steps_to_solve, 0)}</td>
        <td>${fmtMeanStd(m.final_avg_reward_last20)}</td>
        <td>${fmtMeanStd(m.best_episode_reward, 0)}</td>
        <td>${fmtMeanStd(m.training_time_sec)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function stopPlayback() {
    state.playing = false;
    playBtn.textContent = "▶";
    if (state.timerId) {
      clearInterval(state.timerId);
      state.timerId = null;
    }
  }

  function startPlayback() {
    if (state.frame >= state.maxFrame) state.frame = 0;
    state.playing = true;
    playBtn.textContent = "⏸";
    const delay = BASE_DELAY_MS * Number(speedSelect.value);
    state.timerId = setInterval(() => {
      state.frame += 1;
      if (state.frame >= state.maxFrame) {
        state.frame = state.maxFrame;
        slider.value = String(state.frame);
        render();
        stopPlayback();
        return;
      }
      slider.value = String(state.frame);
      render();
    }, delay);
  }

  playBtn.addEventListener("click", () => (state.playing ? stopPlayback() : startPlayback()));
  resetBtn.addEventListener("click", () => {
    stopPlayback();
    state.frame = 0;
    slider.value = "0";
    render();
  });
  slider.addEventListener("input", (e) => {
    stopPlayback();
    state.frame = Number(e.target.value);
    render();
  });
  speedSelect.addEventListener("change", () => {
    if (state.playing) {
      stopPlayback();
      startPlayback();
    }
  });
  container.querySelectorAll(".series-toggle").forEach((box) => {
    box.addEventListener("change", () => {
      const key = box.dataset.series;
      if (box.checked) state.visible.add(key);
      else state.visible.delete(key);
      render();
    });
  });
  rawToggle.addEventListener("change", (e) => {
    state.showRaw = e.target.checked;
    render();
  });

  renderTable();
  render();
}

// ---------- per-algorithm baseline vs improved ----------
function buildCompareTab() {
  const host = document.getElementById("compare-blocks");
  host.innerHTML = "";

  LABELS_ORDER.forEach((key) => {
    const baseline = DATA.runs.find((r) => r.key === key && r.variant === "baseline");
    const improved = DATA.runs.find((r) => r.key === key && r.variant === "improved");
    if (!baseline || !improved) return;
    const color = COLORS[key];

    const block = document.createElement("div");
    block.className = "panel compare-block";
    block.innerHTML = `
      <h3><span style="color:${color};">●</span> ${key}</h3>
      <div class="compare-legend">
        <span><span class="legend-swatch dashed" style="color:${color};"></span>${baseline.label}</span>
        <span><span class="legend-swatch" style="background:${color};"></span>${improved.label}</span>
      </div>
      <canvas class="reward-chart" width="1000" height="300"></canvas>
      <div class="table-wrap" style="margin-top:14px;">
        <table>
          <thead>
            <tr><th>Chỉ số</th><th>${baseline.label}</th><th>${improved.label}</th></tr>
          </thead>
          <tbody>
            <tr><td>Tỷ lệ hội tụ (/${DATA.seeds.length} seed)</td><td>${Math.round(baseline.metrics.solved_rate * DATA.seeds.length)}</td><td>${Math.round(improved.metrics.solved_rate * DATA.seeds.length)}</td></tr>
            <tr><td>Episode hội tụ (mean)</td><td>${fmtMeanStd(baseline.metrics.solved_at_episode, 0)}</td><td>${fmtMeanStd(improved.metrics.solved_at_episode, 0)}</td></tr>
            <tr><td>Reward TB 20 ep cuối (mean±std)</td><td>${fmtMeanStd(baseline.metrics.final_avg_reward_last20)}</td><td>${fmtMeanStd(improved.metrics.final_avg_reward_last20)}</td></tr>
            <tr><td>Độ lệch chuẩn 20 ep cuối (mean)</td><td>${fmtMeanStd(baseline.metrics.final_std_reward_last20)}</td><td>${fmtMeanStd(improved.metrics.final_std_reward_last20)}</td></tr>
            <tr><td>Thời gian train (s, mean)</td><td>${fmtMeanStd(baseline.metrics.training_time_sec)}</td><td>${fmtMeanStd(improved.metrics.training_time_sec)}</td></tr>
          </tbody>
        </table>
      </div>
    `;
    host.appendChild(block);

    const canvas = block.querySelector(".reward-chart");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const padding = { left: 40, right: 16, top: 16, bottom: 24 };
    const maxX = Math.max(baseline.moving_avg_mean.length, improved.moving_avg_mean.length);
    const maxY =
      Math.max(
        ...baseline.moving_avg_mean.map((v, i) => v + baseline.moving_avg_std[i]),
        ...improved.moving_avg_mean.map((v, i) => v + improved.moving_avg_std[i]),
        DATA.solved_threshold,
        10
      ) * 1.05;

    drawAxes(ctx, padding, w, h, maxY);
    drawThreshold(ctx, w, h, padding, maxY, DATA.solved_threshold);
    plotBand(ctx, baseline.moving_avg_mean, baseline.moving_avg_std, color, w, h, padding, maxX, maxY, 0.1);
    plotBand(ctx, improved.moving_avg_mean, improved.moving_avg_std, color, w, h, padding, maxX, maxY, 0.14);
    plotLine(ctx, baseline.moving_avg_mean, color, w, h, padding, maxX, maxY, { alpha: 0.75, dashed: true });
    plotLine(ctx, improved.moving_avg_mean, color, w, h, padding, maxX, maxY, { alpha: 1 });
  });
}

loadData();
