const COLORS = {
  DQN: "#2f6fed",
  REINFORCE: "#e0663e",
  "Actor-Critic": "#1fa774",
};

const BASE_DELAY_MS = 20; // ms per episode at 1x speed multiplier

const state = {
  data: null,
  visible: new Set(["DQN", "REINFORCE", "Actor-Critic"]),
  showRaw: false,
  frame: 0, // last episode index currently shown (playback cursor)
  maxFrame: 0,
  playing: false,
  timerId: null,
};

const canvas = document.getElementById("reward-chart");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("chart-status");
const playBtn = document.getElementById("play-btn");
const resetBtn = document.getElementById("reset-btn");
const slider = document.getElementById("episode-slider");
const episodeLabel = document.getElementById("episode-label");
const speedSelect = document.getElementById("speed-select");
const liveStatsEl = document.getElementById("live-stats");

function seriesKey(run) {
  if (run.name.includes("DQN")) return "DQN";
  if (run.name.includes("REINFORCE")) return "REINFORCE";
  return "Actor-Critic";
}

async function loadData() {
  try {
    const res = await fetch("./data/results.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
    statusEl.textContent = `Môi trường ${state.data.environment} · ${state.data.n_episodes} episodes · seed ${state.data.seed}`;

    state.maxFrame = Math.max(...state.data.runs.map((r) => r.rewards.length)) - 1;
    slider.max = String(state.maxFrame);
    state.frame = state.maxFrame; // mặc định hiện toàn bộ quá trình đã train xong
    slider.value = String(state.frame);

    buildLiveStatsShells();
    renderTable();
    render();
  } catch (err) {
    statusEl.textContent =
      "Không tải được data/results.json — hãy chạy backend/train.py trước, rồi mở trang này qua một local server " +
      "(vd: `py -3.11 -m http.server` trong thư mục frontend) thay vì mở file trực tiếp.";
    console.error(err);
  }
}

function drawAxes(padding, w, h, maxY) {
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
    ctx.strokeStyle = "#e3e6ee33";
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }
}

function plotLine(points, color, w, h, padding, maxX, maxY, alpha = 1) {
  if (!points.length) return;
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = alpha === 1 ? 2 : 1;
  ctx.beginPath();
  points.forEach((v, i) => {
    const x = padding.left + ((w - padding.left - padding.right) * i) / (maxX - 1 || 1);
    const y = h - padding.bottom - ((h - padding.top - padding.bottom) * v) / maxY;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.globalAlpha = 1;

  // chấm đánh dấu episode hiện tại (đầu mút đường vẽ)
  const lastIdx = points.length - 1;
  const x = padding.left + ((w - padding.left - padding.right) * lastIdx) / (maxX - 1 || 1);
  const y = h - padding.bottom - ((h - padding.top - padding.bottom) * points[lastIdx]) / maxY;
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, alpha === 1 ? 4 : 2.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
}

function render() {
  if (!state.data) return;
  const w = canvas.width;
  const h = canvas.height;
  const padding = { left: 40, right: 16, top: 16, bottom: 24 };
  const upto = state.frame + 1; // số episode hiển thị tính đến con trỏ playback

  ctx.clearRect(0, 0, w, h);

  const allRuns = state.data.runs;
  const runs = allRuns.filter((r) => state.visible.has(seriesKey(r)));
  const maxX = state.maxFrame + 1;
  const maxY = Math.max(...allRuns.flatMap((r) => r.rewards), state.data.solved_threshold, 10) * 1.05;

  drawAxes(padding, w, h, maxY);

  const yThresh = h - padding.bottom - ((h - padding.top - padding.bottom) * state.data.solved_threshold) / maxY;
  ctx.strokeStyle = "#8892a6";
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(padding.left, yThresh);
  ctx.lineTo(w - padding.right, yThresh);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#8892a6";
  ctx.fillText(`ngưỡng hội tụ (${state.data.solved_threshold})`, padding.left + 8, yThresh - 6);

  runs.forEach((run) => {
    const key = seriesKey(run);
    const color = COLORS[key];
    const rawSlice = run.rewards.slice(0, upto);
    const avgSlice = run.moving_avg.slice(0, upto);
    if (state.showRaw) plotLine(rawSlice, color, w, h, padding, maxX, maxY, 0.25);
    plotLine(avgSlice, color, w, h, padding, maxX, maxY, 1);
  });

  episodeLabel.textContent = `episode ${state.frame} / ${state.maxFrame}`;
  updateLiveStats();
}

function buildLiveStatsShells() {
  liveStatsEl.innerHTML = "";
  state.data.runs.forEach((run) => {
    const key = seriesKey(run);
    const card = document.createElement("div");
    card.className = "live-card";
    card.style.setProperty("--stat-color", COLORS[key]);
    card.dataset.series = key;
    card.innerHTML = `
      <div class="live-name"><span>${run.name}</span><span class="badge-solved" hidden>hội tụ</span></div>
      <div class="live-reward">—</div>
      <div class="live-sub">reward moving-avg</div>
    `;
    liveStatsEl.appendChild(card);
  });
}

function updateLiveStats() {
  state.data.runs.forEach((run) => {
    const key = seriesKey(run);
    const card = liveStatsEl.querySelector(`[data-series="${key}"]`);
    if (!card) return;
    const idx = Math.min(state.frame, run.moving_avg.length - 1);
    const avgVal = run.moving_avg[idx];
    card.querySelector(".live-reward").textContent = avgVal.toFixed(1);
    const solvedNow = run.metrics.solved_at_episode !== null && idx >= run.metrics.solved_at_episode;
    card.querySelector(".badge-solved").hidden = !solvedNow;
    card.style.opacity = state.visible.has(key) ? "1" : "0.4";
  });
}

function renderTable() {
  const tbody = document.querySelector("#metrics-table tbody");
  tbody.innerHTML = "";
  state.data.runs.forEach((run) => {
    const m = run.metrics;
    const key = seriesKey(run);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span style="color:${COLORS[key]}; font-weight:600;">●</span> ${run.name}</td>
      <td>${run.family}</td>
      <td>${m.solved_at_episode ?? "chưa hội tụ"}</td>
      <td>${m.env_steps_to_solve ?? "—"}</td>
      <td>${m.final_avg_reward_last20.toFixed(1)}</td>
      <td>${m.final_std_reward_last20.toFixed(1)}</td>
      <td>${m.best_episode_reward.toFixed(0)}</td>
      <td>${m.training_time_sec.toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function stopPlayback() {
  state.playing = false;
  playBtn.textContent = "▶";
  playBtn.setAttribute("aria-label", "Phát");
  if (state.timerId) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
}

function startPlayback() {
  if (state.frame >= state.maxFrame) state.frame = 0; // phát lại từ đầu nếu đang ở cuối
  state.playing = true;
  playBtn.textContent = "⏸";
  playBtn.setAttribute("aria-label", "Tạm dừng");

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

playBtn.addEventListener("click", () => {
  if (state.playing) stopPlayback();
  else startPlayback();
});

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

document.querySelectorAll(".series-toggle").forEach((box) => {
  box.addEventListener("change", () => {
    const key = box.dataset.series;
    if (box.checked) state.visible.add(key);
    else state.visible.delete(key);
    render();
  });
});

document.getElementById("show-raw").addEventListener("change", (e) => {
  state.showRaw = e.target.checked;
  render();
});

loadData();
