// Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The page. It knows three things: which robot is selected, what that robot said it can do,
// and what has happened. Everything else it asks the server, which asks the mesh.
//
// The one rule worth stating: this file never decides what a robot can do. `capabilities()`
// is an RPC and its answer drives which controls are enabled and which warnings appear, so a
// Lite3 and a Go2 get different pages without this file knowing what either of them is.

const $ = (id) => document.getElementById(id);

const state = {
  deviceId: null,
  capabilities: null,
  busy: false,
  paused: false,
};

// ── plumbing ────────────────────────────────────────────────────────────────
async function invoke(fn, params = {}) {
  if (!state.deviceId) return { ok: false, error: "no robot selected" };
  const response = await fetch("/api/invoke", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_id: state.deviceId, function: fn, params }),
  });
  const body = await response.json().catch(() => ({ ok: false, error: "bad response" }));
  if (!body.ok) return { ok: false, error: body.error || "request failed" };
  // The transport succeeded; the device may still have refused. Unwrap one layer so callers
  // see the driver's own {ok, error} rather than the envelope's.
  const result = body.result || {};
  if (result.success === false) return { ok: false, error: describeError(result.error) };
  return result.result !== undefined ? result.result : result;
}

function describeError(error) {
  if (!error) return "the device reported a failure with no message";
  if (typeof error === "string") return error;
  return error.message || JSON.stringify(error);
}

function show(el, value, isError) {
  el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  el.className = "result " + (isError ? "bad" : "ok");
}

// ── devices ─────────────────────────────────────────────────────────────────
async function refreshDevices() {
  let devices = [];
  try {
    const body = await (await fetch("/api/devices")).json();
    devices = body.ok ? body.devices : [];
    setBadge($("mesh-badge"), body.ok ? "mesh up" : "mesh down", body.ok ? "on" : "hot");
  } catch {
    setBadge($("mesh-badge"), "mesh down", "hot");
  }

  const select = $("device-select");
  const previous = state.deviceId;
  select.innerHTML = "";
  if (!devices.length) {
    select.innerHTML = '<option value="">no robots found</option>';
    state.deviceId = null;
    renderCapabilities(null);
    return;
  }
  for (const device of devices) {
    const option = document.createElement("option");
    option.value = device.device_id;
    option.textContent = `${device.device_id} · ${device.device_type || "device"}`;
    select.appendChild(option);
  }
  // Keep the operator's selection across a poll. Re-selecting device one every ten seconds
  // while someone is driving device two is the kind of bug that only shows up with a robot.
  state.deviceId = devices.some((d) => d.device_id === previous) ? previous : devices[0].device_id;
  select.value = state.deviceId;
  if (state.deviceId !== previous) await onDeviceChanged();
}

async function onDeviceChanged() {
  const capabilities = await invoke("get_capabilities");
  renderCapabilities(capabilities.ok === false ? null : capabilities);
  await refreshModels();
}

function setBadge(el, text, cls) {
  el.textContent = text;
  el.className = "badge " + (cls || "muted");
}

// ── capabilities drive the controls ─────────────────────────────────────────
function renderCapabilities(caps) {
  state.capabilities = caps;
  const notes = $("capability-notes");
  notes.innerHTML = "";

  if (!caps) {
    setBadge($("platform-badge"), "—", "muted");
    setBadge($("motion-badge"), "motion —", "muted");
    $("safety").classList.add("hidden");
    setPadEnabled(false);
    return;
  }

  // The ceiling is the robot's, not the page's. Reading it from the device means a driver
  // that lowers MAX_SECONDS lowers the input too, instead of the form quietly permitting a
  // number the worker will silently clamp.
  if (caps.max_seconds) $("seconds").max = caps.max_seconds;
  setBadge($("platform-badge"), caps.platform || "?", "off");
  setBadge($("motion-badge"),
    caps.motion_enabled ? "motion enabled" : "motion disabled",
    caps.motion_enabled ? "hot" : "off");
  $("safety").classList.toggle("hidden", !caps.motion_enabled);
  setPadEnabled(!!caps.motion_enabled);

  if (!caps.motion_enabled) {
    addNote(notes, "warn",
      "<strong>This device was started without <code>--allow-motion</code>.</strong> " +
      "It is status-and-checkpoints only; the motion keys are disabled. Restart the driver " +
      "with the flag, with an operator on the abort, to enable them.");
  }

  // An axis with no measured gait floor is the honest version of a greyed-out button: the
  // control works, and pressing it may do nothing at all for a reason that is not a fault.
  for (const axis of caps.unmeasured_axes || []) {
    addNote(notes, "warn",
      `<strong>No ${axis} gait floor has been measured on the ${caps.platform}.</strong> ` +
      `A command on this axis may produce no movement at all, and that would not be a ` +
      `fault — see issue #42. The result panel reports what actually moved.`);
  }

  if (caps.lie_down_changes_posture === false) {
    addNote(notes, "warn", `<strong>Lie down does not lie this robot down.</strong> ${caps.posture_note}`);
  }
  addNote(notes, "warn",
    `<strong>Reverse is open-loop into unobserved space.</strong> ${caps.reverse_note}.`);
}

function addNote(parent, kind, html) {
  const div = document.createElement("div");
  div.className = "note " + kind;
  div.innerHTML = html;
  parent.appendChild(div);
}

function setPadEnabled(enabled) {
  for (const key of document.querySelectorAll(".key")) {
    // Stop is never gated. If motion is disabled it will be refused by the device anyway,
    // but a stop button that cannot be pressed is the wrong affordance on a robot console.
    key.disabled = key.dataset.fn === "stop" ? false : !enabled;
  }
}

// ── motion ──────────────────────────────────────────────────────────────────
function motionParams(fn) {
  const seconds = parseFloat($("seconds").value);
  const speed = parseFloat($("speed").value);
  const rate = parseFloat($("rate").value);
  const force = $("force").checked;
  if (fn === "turn_left" || fn === "turn_right") return { seconds, rate_rad_s: rate };
  if (fn === "walk_back") return { seconds, speed_mps: speed };
  if (fn === "walk_forward" || fn === "strafe_left" || fn === "strafe_right") {
    return { seconds, speed_mps: speed, force };
  }
  return {};
}

async function sendMotion(button) {
  const fn = button.dataset.fn;
  // Stop bypasses the busy interlock deliberately: the one command you must be able to send
  // while the robot is already doing something is the one that makes it stop.
  if (state.busy && fn !== "stop") return;
  if (fn !== "stop") { state.busy = true; button.classList.add("busy"); }
  try {
    const result = await invoke(fn, motionParams(fn));
    show($("motion-result"), summariseMotion(fn, result), result.ok === false);
  } finally {
    state.busy = false;
    button.classList.remove("busy");
  }
}

function summariseMotion(fn, result) {
  if (result.ok === false) return `${fn} refused\n\n${result.error}`;
  const lines = [`${fn} ok`];
  if (result.travelled_m !== null && result.travelled_m !== undefined) {
    lines.push(`travelled      ${result.travelled_m} m`);
  }
  if (result.turned_deg) lines.push(`turned         ${result.turned_deg}°`);
  if (result.seconds !== undefined) lines.push(`held           ${result.seconds} s`);
  if (result.delivered_fraction !== undefined) {
    lines.push(`delivered      ${result.delivered_fraction} of commanded`);
  }
  if (result.note) lines.push(`\n${result.note}`);
  if (result.warning) lines.push(`\n⚠ ${result.warning}`);
  return lines.join("\n");
}

// ── checkpoints ─────────────────────────────────────────────────────────────
async function refreshModels() {
  const body = $("model-table").querySelector("tbody");
  const result = await invoke("list_models");
  if (result.ok === false) {
    body.innerHTML = `<tr><td colspan="6" class="empty">${escapeHtml(result.error)}</td></tr>`;
    return;
  }
  const models = result.models || [];
  $("disk-note").textContent = result.free_bytes
    ? `${formatBytes(result.free_bytes)} free · config ${result.config_path}`
    : "";
  if (!models.length) {
    body.innerHTML = '<tr><td colspan="6" class="empty">no checkpoints on this robot</td></tr>';
    return;
  }
  body.innerHTML = "";
  for (const model of models) body.appendChild(modelRow(model));
}

function modelRow(model) {
  const tr = document.createElement("tr");
  if (model.active) tr.className = "armed";

  const problems = model.problems || [];
  let pill = '<span class="pill armed">armed</span>';
  if (!model.active) {
    pill = model.loadable && model.compatible_with_config
      ? '<span class="pill ready">ready</span>'
      : '<span class="pill bad">incompatible</span>';
  }

  tr.innerHTML = `
    <td class="name">${escapeHtml(model.name)}</td>
    <td class="num">${formatBytes(model.size_bytes)}</td>
    <td class="num">${model.rays ?? "—"}</td>
    <td class="num">${model.trained_lidar_range_vmas ?? "—"}</td>
    <td>${pill}</td>
    <td class="actions"></td>`;

  const actions = tr.querySelector(".actions");
  if (!model.active) {
    const arm = button("Arm", "btn tiny", async () => {
      const result = await invoke("select_model", { name: model.name });
      show($("model-result"), result.ok === false ? result.error : result, result.ok === false);
      await refreshModels();
    });
    arm.disabled = !(model.loadable && model.compatible_with_config);
    if (arm.disabled && problems.length) arm.title = problems.join(" ");
    actions.appendChild(arm);

    actions.appendChild(button("Unload", "btn ghost tiny", async () => {
      const result = await invoke("delete_model", { name: model.name });
      show($("model-result"), result.ok === false ? result.error : result, result.ok === false);
      await refreshModels();
    }));
  }
  if (problems.length) tr.title = problems.join(" ");
  return tr;
}

function button(label, className, onClick) {
  const el = document.createElement("button");
  el.textContent = label;
  el.className = className;
  el.addEventListener("click", onClick);
  return el;
}

async function downloadModel() {
  const source = $("source").value.trim();
  if (!source) return show($("model-result"), "give a source first", true);
  const params = { source };
  const name = $("install-as").value.trim();
  if (name) params.name = name;

  $("download").disabled = true;
  show($("model-result"), `fetching ${source}…`, false);
  try {
    const result = await invoke("download_model", params);
    if (result.ok === false) {
      show($("model-result"), result.error, true);
    } else {
      const model = result.model || {};
      show($("model-result"),
        `loaded ${model.name}\n` +
        `bytes          ${result.downloaded_bytes}\n` +
        `sha256         ${model.sha256}\n` +
        `rays           ${model.rays ?? "—"}\n` +
        `trained range  ${model.trained_lidar_range_vmas ?? "—"}\n` +
        `runnable now   ${model.compatible_with_config ? "yes" : "NO — " + (model.problems || []).join(" ")}\n\n` +
        `Not armed. Arm it in the table above when you want the next run to use it.`, false);
      await refreshModels();
    }
  } finally {
    $("download").disabled = false;
  }
}

async function browseBucket() {
  const bucket = $("bucket").value.trim();
  if (!bucket) return show($("model-result"), "give a bucket name first", true);
  const list = $("cloud-list");
  list.innerHTML = "";
  const result = await invoke("list_cloud_models", { bucket, prefix: $("prefix").value.trim() });
  if (result.ok === false) return show($("model-result"), result.error, true);
  const objects = result.objects || [];
  if (!objects.length) {
    return addNote(list, "", `no <code>.npz</code> objects under <code>${escapeHtml(bucket)}</code>`);
  }
  for (const object of objects) {
    const div = document.createElement("div");
    div.className = "note";
    div.innerHTML = `<strong>${escapeHtml(object.key)}</strong> · ${formatBytes(object.size_bytes)} · ${object.last_modified || ""} `;
    div.appendChild(button("Use", "btn ghost tiny", () => { $("source").value = object.uri; }));
    list.appendChild(div);
  }
}

// ── events ──────────────────────────────────────────────────────────────────
function startEventStream() {
  const source = new EventSource("/api/events");
  const list = $("events");
  source.onmessage = (message) => {
    if (state.paused) return;
    let record;
    try { record = JSON.parse(message.data); } catch { return; }
    if (!passesFilter(record)) return;
    if (list.querySelector(".empty")) list.innerHTML = "";
    // Newest first. An operator watching a robot reads the top of the list, and a stream
    // that appends means the interesting line walks off the bottom of the box.
    list.prepend(eventRow(record));
    while (list.childElementCount > 400) list.lastElementChild.remove();
  };
  source.onerror = () => setBadge($("mesh-badge"), "stream lost", "hot");
  source.onopen = () => setBadge($("mesh-badge"), "mesh up", "on");
}

function passesFilter(record) {
  if ($("only-selected").checked && record.device_id !== state.deviceId) return false;
  const needle = $("event-filter").value.trim().toLowerCase();
  if (!needle) return true;
  return (record.event + " " + record.device_id + " " + JSON.stringify(record.payload))
    .toLowerCase().includes(needle);
}

function eventRow(record) {
  const li = document.createElement("li");
  li.className = record.event;
  const when = new Date((record.received || 0) * 1000).toLocaleTimeString();
  li.innerHTML =
    `<span class="t">${when}</span>` +
    `<span class="d">${escapeHtml(record.device_id)}</span>` +
    `<span class="e">${escapeHtml(record.event)}</span>` +
    `<span class="p">${escapeHtml(summarisePayload(record.payload))}</span>`;
  return li;
}

function summarisePayload(payload) {
  if (!payload || !Object.keys(payload).length) return "";
  return Object.entries(payload)
    .map(([key, value]) => {
      if (value === null || value === "" ) return null;
      if (typeof value === "object") return `${key}=${JSON.stringify(value)}`;
      return `${key}=${value}`;
    })
    .filter(Boolean)
    .join("  ");
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes, unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

// ── wiring ──────────────────────────────────────────────────────────────────
function init() {
  for (const key of document.querySelectorAll(".key")) {
    key.addEventListener("click", () => sendMotion(key));
  }
  $("device-select").addEventListener("change", (event) => {
    state.deviceId = event.target.value || null;
    onDeviceChanged();
  });
  $("download").addEventListener("click", downloadModel);
  $("browse").addEventListener("click", browseBucket);
  $("pause").addEventListener("click", () => {
    state.paused = !state.paused;
    $("pause").textContent = state.paused ? "Resume" : "Pause";
  });
  $("clear").addEventListener("click", () => { $("events").innerHTML = ""; });

  startEventStream();
  refreshDevices();
  // Devices come and go — a robot rebooted mid-demo should reappear without a page reload.
  setInterval(refreshDevices, 10000);
}

document.addEventListener("DOMContentLoaded", init);
