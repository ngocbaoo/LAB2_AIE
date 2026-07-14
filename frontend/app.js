const COLOR = "#16a34a"; // Actor-Critic accent (single algorithm now)
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
  buildTrainingView(document.getElementById("training"));
}

// ---------- shared drawing helpers ----------
// Series (P&L) can go negative, so every helper takes an explicit [minY, maxY]
// range instead of assuming 0 as the floor.
function drawAxes(ctx, padding, w, h, minY, maxY) {
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
    const val = minY + ((maxY - minY) / ySteps) * i;
    const y = h - padding.bottom - ((h - padding.top - padding.bottom) * i) / ySteps;
    ctx.fillText(val.toFixed(1), 6, y + 4);
    ctx.strokeStyle = "#8892a61a";
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }

  if (minY < 0 && maxY > 0) {
    const [, yZero] = toXY(0, 0, w, h, padding, 2, minY, maxY);
    ctx.strokeStyle = "#8892a655";
    ctx.beginPath();
    ctx.moveTo(padding.left, yZero);
    ctx.lineTo(w - padding.right, yZero);
    ctx.stroke();
  }
}

function toXY(i, v, w, h, padding, maxX, minY, maxY) {
  const x = padding.left + ((w - padding.left - padding.right) * i) / (maxX - 1 || 1);
  const range = maxY - minY || 1;
  const y = h - padding.bottom - ((h - padding.top - padding.bottom) * (v - minY)) / range;
  return [x, y];
}

function plotLine(ctx, points, color, w, h, padding, maxX, minY, maxY, { alpha = 1, dashed = false, dot = true } = {}) {
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = alpha >= 1 ? 2 : 1;
  if (dashed) ctx.setLineDash([6, 4]);
  ctx.beginPath();
  points.forEach((v, i) => {
    const [x, y] = toXY(i, v, w, h, padding, maxX, minY, maxY);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  if (dot) {
    const lastIdx = points.length - 1;
    const [x, y] = toXY(lastIdx, points[lastIdx], w, h, padding, maxX, minY, maxY);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, alpha >= 1 ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function plotBand(ctx, meanArr, stdArr, color, w, h, padding, maxX, minY, maxY, alpha) {
  if (!meanArr.length) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  meanArr.forEach((v, i) => {
    const [x, y] = toXY(i, v + stdArr[i], w, h, padding, maxX, minY, maxY);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  for (let i = meanArr.length - 1; i >= 0; i--) {
    const [x, y] = toXY(i, meanArr[i] - stdArr[i], w, h, padding, maxX, minY, maxY);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawThreshold(ctx, w, h, padding, minY, maxY, threshold) {
  const [, y] = toXY(0, threshold, w, h, padding, 2, minY, maxY);
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
  ctx.fillText(`ngưỡng hội tụ (${threshold}%)`, padding.left + 8, y - 6);
  ctx.restore();
}

function fmtMeanStd(obj, decimals = 1) {
  if (!obj || obj.mean === null || obj.mean === undefined) return "chưa hội tụ";
  if (obj.std === null || obj.n <= 1) return obj.mean.toFixed(decimals);
  return `${obj.mean.toFixed(decimals)} ± ${obj.std.toFixed(decimals)}`;
}

// ---------- baseline vs improved playback view (single algorithm) ----------
function buildTrainingView(container) {
  const runs = DATA.runs; // [baseline, improved]
  const byVariant = Object.fromEntries(runs.map((r) => [r.variant, r]));
  const maxFrame = Math.max(...runs.map((r) => r.moving_avg_mean.length)) - 1;

  const state = {
    visible: new Set(["baseline", "improved"]),
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
    const card = document.createElement("div");
    card.className = "live-card";
    card.style.setProperty("--stat-color", COLOR);
    card.dataset.series = run.variant;
    card.innerHTML = `
      <div class="live-name"><span>${run.label}</span><span class="badge-solved" hidden>hội tụ</span></div>
      <div class="live-reward">—</div>
      <div class="live-sub">reward moving-avg (mean qua seed, %)</div>
    `;
    liveStatsEl.appendChild(card);
  });

  function render() {
    const w = canvas.width;
    const h = canvas.height;
    const padding = { left: 40, right: 16, top: 16, bottom: 24 };
    const upto = state.frame + 1;

    ctx.clearRect(0, 0, w, h);

    const visibleRuns = runs.filter((r) => state.visible.has(r.variant));
    const maxX = state.maxFrame + 1;
    const highs = runs.flatMap((r) => r.moving_avg_mean.map((v, i) => v + r.moving_avg_std[i]));
    const lows = runs.flatMap((r) => r.moving_avg_mean.map((v, i) => v - r.moving_avg_std[i]));
    const maxY = Math.max(...highs, DATA.solved_threshold, 1) * 1.1;
    const minY = Math.min(...lows, 0) * 1.1;

    drawAxes(ctx, padding, w, h, minY, maxY);
    drawThreshold(ctx, w, h, padding, minY, maxY, DATA.solved_threshold);

    visibleRuns.forEach((run) => {
      const dashed = run.variant === "baseline";
      const meanSlice = run.moving_avg_mean.slice(0, upto);
      const stdSlice = run.moving_avg_std.slice(0, upto);
      plotBand(ctx, meanSlice, stdSlice, COLOR, w, h, padding, maxX, minY, maxY, dashed ? 0.08 : 0.14);
      if (state.showRaw) {
        plotLine(ctx, run.rewards_mean.slice(0, upto), COLOR, w, h, padding, maxX, minY, maxY, { alpha: 0.25, dot: false, dashed });
      }
      plotLine(ctx, meanSlice, COLOR, w, h, padding, maxX, minY, maxY, { alpha: dashed ? 0.75 : 1, dashed });
    });

    episodeLabel.textContent = `episode ${state.frame} / ${state.maxFrame}`;

    runs.forEach((run) => {
      const card = liveStatsEl.querySelector(`[data-series="${run.variant}"]`);
      if (!card) return;
      const idx = Math.min(state.frame, run.moving_avg_mean.length - 1);
      card.querySelector(".live-reward").textContent = run.moving_avg_mean[idx].toFixed(1);
      const solvedMean = run.metrics.solved_at_episode.mean;
      const solvedNow = solvedMean !== null && idx >= solvedMean;
      card.querySelector(".badge-solved").hidden = !solvedNow;
      card.style.opacity = state.visible.has(run.variant) ? "1" : "0.4";
    });
  }

  function renderTable() {
    const tbody = container.querySelector(".metrics-table tbody");
    tbody.innerHTML = "";
    runs.forEach((run) => {
      const m = run.metrics;
      const dashed = run.variant === "baseline";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span style="color:${COLOR}; font-weight:600;">${dashed ? "◌" : "●"}</span> ${run.label}</td>
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

loadData();
