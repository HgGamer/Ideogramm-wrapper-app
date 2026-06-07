/* Ideogram 4 Studio — renderer logic */
const $ = (id) => document.getElementById(id);

const viewSetup = $('view-setup');
const viewStudio = $('view-studio');

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  const forced = new URLSearchParams(location.search).get('view');
  if (forced === 'studio') { showStudio(); return; }
  if (forced === 'setup') { viewSetup.classList.remove('hidden'); return; }
  const state = await window.studio.setupState();
  if (state.setupComplete && state.hfToken) {
    showStudio();
  } else {
    viewSetup.classList.remove('hidden');
  }
}
boot();

// external links
document.addEventListener('click', (e) => {
  const a = e.target.closest('a[data-url]');
  if (a) { e.preventDefault(); window.studio.openExternal(a.dataset.url); }
});

// ---------------------------------------------------------------------------
// SETUP
// ---------------------------------------------------------------------------
const setupLog = $('setup-log');
const setupLed = $('setup-led');
const setupErr = $('setup-error');
const btnSetup = $('btn-setup');

function appendLog(pane, text) {
  if (pane.dataset.dirty !== '1') { pane.textContent = ''; pane.dataset.dirty = '1'; }
  pane.textContent += text;
  pane.scrollTop = pane.scrollHeight;
}

window.studio.onSetupLog((t) => appendLog(setupLog, t));
window.studio.onSetupStep(({ id, status }) => {
  const li = document.querySelector(`.step[data-id="${id}"]`);
  if (!li) return;
  li.classList.remove('active', 'done', 'error');
  li.classList.add(status);
  li.querySelector('.step-state').textContent =
    status === 'active' ? 'RUNNING' : status === 'done' ? 'OK' : 'FAILED';
});
window.studio.onSetupError((msg) => {
  const active = document.querySelector('.step.active');
  if (active) {
    active.classList.replace('active', 'error');
    active.querySelector('.step-state').textContent = 'FAILED';
  }
  setupLed.className = 'log-led err';
  setupErr.textContent = msg;
  setupErr.classList.remove('hidden');
  btnSetup.disabled = false;
  btnSetup.textContent = 'RETRY SETUP ↦';
});
window.studio.onSetupDone(() => {
  setupLed.className = 'log-led ok';
  setTimeout(showStudio, 700);
});

btnSetup.addEventListener('click', async () => {
  const hf = $('setup-hf').value.trim();
  if (!hf) {
    setupErr.textContent = 'A Hugging Face token is required — the model weights are gated.';
    setupErr.classList.remove('hidden');
    return;
  }
  await window.studio.setKeys({ hfToken: hf });
  setupErr.classList.add('hidden');
  btnSetup.disabled = true;
  btnSetup.textContent = 'WORKING…';
  setupLed.className = 'log-led live';
  setupLog.dataset.dirty = '0';
  await window.studio.runSetup();
  btnSetup.disabled = false;
  btnSetup.textContent = 'BEGIN SETUP ↦';
});

// ---------------------------------------------------------------------------
// STUDIO
// ---------------------------------------------------------------------------
const genLog = $('gen-log');
const genLed = $('gen-led');
const genErr = $('gen-error');
const btnGen = $('btn-generate');
const btnCancel = $('btn-cancel');
const canvasImg = $('canvas-img');
const canvasEmpty = $('canvas-empty');
const canvasWorking = $('canvas-working');

async function showStudio() {
  viewSetup.classList.add('hidden');
  viewStudio.classList.remove('hidden');
  // populate sampler presets from run_inference.py --help
  const presets = await window.studio.presets();
  const sel = $('preset');
  sel.innerHTML = '';
  for (const p of presets) {
    const o = document.createElement('option');
    o.value = o.textContent = p;
    if (p === 'V4_QUALITY_48') o.selected = true;
    sel.appendChild(o);
  }
  refreshHistory();
}

window.studio.onGenLog((t) => appendLog(genLog, t));

function showImage(p) {
  canvasImg.src = `file://${p}?${Date.now()}`;
  canvasImg.classList.remove('hidden');
  canvasEmpty.classList.add('hidden');
  document.querySelectorAll('.strip img').forEach((i) => i.classList.toggle('selected', i.dataset.path === p));
}

async function refreshHistory(selectPath) {
  const items = await window.studio.history();
  const strip = $('history');
  strip.innerHTML = '';
  if (!items.length) {
    strip.innerHTML = '<span class="strip-empty">— EMPTY —</span>';
    return;
  }
  for (const it of items) {
    const img = document.createElement('img');
    img.src = `file://${it.path}`;
    img.dataset.path = it.path;
    img.title = it.path.split('/').pop();
    img.addEventListener('click', () => showImage(it.path));
    if (it.path === selectPath) img.classList.add('selected');
    strip.appendChild(img);
  }
}

btnGen.addEventListener('click', async () => {
  const prompt = $('prompt').value.trim();
  if (!prompt) { $('prompt').focus(); return; }
  genErr.classList.add('hidden');
  btnGen.disabled = true;
  btnGen.textContent = 'DEVELOPING…';
  btnCancel.classList.remove('hidden');
  canvasWorking.classList.remove('hidden');
  genLed.className = 'log-led live';
  genLog.dataset.dirty = '0';

  const result = await window.studio.generate({
    prompt,
    width: clamp16($('width')),
    height: clamp16($('height')),
    seed: parseInt($('seed').value, 10) || 0,
    preset: $('preset').value,
  });

  btnGen.disabled = false;
  btnGen.textContent = 'EXPOSE ◉';
  btnCancel.classList.add('hidden');
  canvasWorking.classList.add('hidden');

  if (result.ok) {
    genLed.className = 'log-led ok';
    showImage(result.image);
    refreshHistory(result.image);
  } else if (result.cancelled) {
    genLed.className = 'log-led';
    appendLog(genLog, '\n— aborted by user —\n');
  } else {
    genLed.className = 'log-led err';
    genErr.textContent = result.error || 'Generation failed.';
    genErr.classList.remove('hidden');
  }
});

btnCancel.addEventListener('click', () => window.studio.cancel());

// dimensions must be multiples of 16, 256–2048
function clamp16(input) {
  let v = parseInt(input.value, 10) || 1024;
  v = Math.max(256, Math.min(2048, Math.round(v / 16) * 16));
  input.value = v;
  return v;
}

// ⌘↵ to generate
document.addEventListener('keydown', (e) => {
  if (e.metaKey && e.key === 'Enter' && !viewStudio.classList.contains('hidden') && !btnGen.disabled) {
    btnGen.click();
  }
});

$('btn-outputs').addEventListener('click', () => window.studio.openOutputs());

// ---------------------------------------------------------------------------
// settings drawer
// ---------------------------------------------------------------------------
$('btn-settings').addEventListener('click', () => $('settings').classList.remove('hidden'));
$('btn-close-settings').addEventListener('click', () => $('settings').classList.add('hidden'));
$('btn-save-keys').addEventListener('click', async () => {
  const keys = {};
  const hf = $('key-hf').value.trim();
  const ideo = $('key-ideo').value.trim();
  if (hf) keys.hfToken = hf;
  if (ideo) keys.ideogramKey = ideo;
  await window.studio.setKeys(keys);
  $('key-hf').value = ''; $('key-ideo').value = '';
  $('settings').classList.add('hidden');
});
$('settings').addEventListener('click', (e) => {
  if (e.target === $('settings')) $('settings').classList.add('hidden');
});
