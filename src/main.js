const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execFileSync } = require('child_process');

// ---------------------------------------------------------------------------
// Paths & config
// ---------------------------------------------------------------------------
const USER_DATA = app.getPath('userData');
const REPO_URL = 'https://github.com/ideogram-oss/ideogram4';
const REPO_DIR = path.join(USER_DATA, 'ideogram4');
const VENV_DIR = path.join(USER_DATA, 'venv');
const OUT_DIR = path.join(USER_DATA, 'outputs');
const CONFIG_PATH = path.join(USER_DATA, 'config.json');
const PYTHON = path.join(VENV_DIR, 'bin', 'python');

// GUI apps on macOS don't inherit the shell PATH — augment it so we can find
// git / uv / homebrew python.
const EXTRA_PATHS = [
  '/opt/homebrew/bin',
  '/usr/local/bin',
  '/usr/bin',
  '/bin',
  path.join(process.env.HOME || '', '.local', 'bin'),
  path.join(process.env.HOME || '', '.cargo', 'bin'),
];
const ENV_PATH = [...new Set([...EXTRA_PATHS, ...(process.env.PATH || '').split(':')])].join(':');

function loadConfig() {
  try { return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')); } catch { return {}; }
}
function saveConfig(cfg) {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2));
}

function which(bin) {
  try {
    return execFileSync('/usr/bin/which', [bin], { env: { ...process.env, PATH: ENV_PATH } })
      .toString().trim() || null;
  } catch { return null; }
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------
let win;
function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 980,
    minHeight: 700,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 18, y: 20 },
    backgroundColor: '#121110',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  const forceView = (process.argv.find((a) => a.startsWith('--view=')) || '').split('=')[1];
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'),
    forceView ? { query: { view: forceView } } : undefined);

  // dev verification: --screenshot=/path.png captures the window and quits
  const shot = process.argv.find((a) => a.startsWith('--screenshot='));
  if (shot) {
    win.webContents.once('did-finish-load', () => {
      setTimeout(async () => {
        const img = await win.webContents.capturePage();
        fs.writeFileSync(shot.split('=')[1], img.toPNG());
        app.quit();
      }, 2500);
    });
  }
}

app.whenReady().then(() => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on('window-all-closed', () => app.quit());

// ---------------------------------------------------------------------------
// Helpers: spawn a step, streaming output to the renderer
// ---------------------------------------------------------------------------
function send(channel, payload) { if (win && !win.isDestroyed()) win.webContents.send(channel, payload); }

function runStep(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      env: { ...process.env, PATH: ENV_PATH, ...opts.env },
      cwd: opts.cwd,
    });
    let tail = '';
    const onData = (d) => {
      const text = d.toString();
      tail = (tail + text).slice(-4000);
      send('setup:log', text);
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${path.basename(cmd)} exited with code ${code}\n${tail.slice(-1500)}`));
    });
  });
}

// ---------------------------------------------------------------------------
// Setup: status + run
// ---------------------------------------------------------------------------
function setupState() {
  const cfg = loadConfig();
  return {
    git: !!which('git'),
    uv: !!which('uv'),
    repo: fs.existsSync(path.join(REPO_DIR, 'run_inference.py')),
    venv: fs.existsSync(PYTHON),
    installed: !!cfg.installed,
    hfToken: !!cfg.hfToken,
    ideogramKey: !!cfg.ideogramKey,
    setupComplete: !!cfg.installed && fs.existsSync(PYTHON) && fs.existsSync(path.join(REPO_DIR, 'run_inference.py')),
    paths: { repo: REPO_DIR, venv: VENV_DIR, outputs: OUT_DIR },
  };
}

ipcMain.handle('setup:state', () => setupState());

ipcMain.handle('setup:run', async () => {
  const cfg = loadConfig();
  const step = (id, status) => send('setup:step', { id, status });
  try {
    // 1 — toolchain
    step('tools', 'active');
    const git = which('git');
    if (!git) throw new Error('git not found. Install Xcode Command Line Tools: xcode-select --install');
    let uv = which('uv');
    if (!uv) {
      send('setup:log', 'uv not found — installing via official installer...\n');
      await runStep('/bin/sh', ['-c', 'curl -LsSf https://astral.sh/uv/install.sh | sh']);
      uv = which('uv');
      if (!uv) throw new Error('uv installation failed. Install manually: https://docs.astral.sh/uv/');
    }
    step('tools', 'done');

    // 2 — clone / update repo
    step('repo', 'active');
    if (fs.existsSync(path.join(REPO_DIR, '.git'))) {
      send('setup:log', 'Repository exists — pulling latest...\n');
      await runStep(git, ['-C', REPO_DIR, 'pull', '--ff-only']);
    } else {
      await runStep(git, ['clone', '--depth', '1', REPO_URL, REPO_DIR]);
    }
    step('repo', 'done');

    // 3 — venv (Python 3.12: new enough for the repo, old enough for torch wheels)
    step('venv', 'active');
    if (!fs.existsSync(PYTHON)) {
      await runStep(uv, ['venv', '--python', '3.12', VENV_DIR]);
    } else {
      send('setup:log', 'Virtual environment already exists — reusing.\n');
    }
    step('venv', 'done');

    // 4 — install package + deps into the venv
    step('deps', 'active');
    await runStep(uv, ['pip', 'install', '--python', PYTHON, REPO_DIR]);
    step('deps', 'done');

    // 5 — verify torch + MPS
    step('verify', 'active');
    await runStep(PYTHON, ['-c', [
      'import torch, platform',
      'print(f"python {platform.python_version()}  torch {torch.__version__}")',
      'print(f"MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}")',
    ].join('\n')]);
    step('verify', 'done');

    cfg.installed = true;
    saveConfig(cfg);
    send('setup:done', setupState());
    return { ok: true };
  } catch (err) {
    send('setup:error', String(err.message || err));
    return { ok: false, error: String(err.message || err) };
  }
});

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------
ipcMain.handle('config:setKeys', (_e, { hfToken, ideogramKey }) => {
  const cfg = loadConfig();
  if (hfToken !== undefined) cfg.hfToken = hfToken.trim();
  if (ideogramKey !== undefined) cfg.ideogramKey = ideogramKey.trim();
  saveConfig(cfg);
  return setupState();
});

// ---------------------------------------------------------------------------
// Sampler presets — parse them out of `run_inference.py --help`
// ---------------------------------------------------------------------------
ipcMain.handle('gen:presets', async () => {
  try {
    const help = execFileSync(PYTHON, [path.join(REPO_DIR, 'run_inference.py'), '--help'], {
      env: { ...process.env, PATH: ENV_PATH }, cwd: REPO_DIR, timeout: 60000,
    }).toString();
    const m = help.match(/--sampler-preset\s*\{([^}]+)\}/);
    if (m) return m[1].split(',').map((s) => s.trim()).filter(Boolean);
  } catch { /* fall through */ }
  return ['V4_QUALITY_48', 'V4_DEFAULT_20', 'V4_TURBO_12'];
});

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------
let genProc = null;

ipcMain.handle('gen:start', async (_e, params) => {
  if (genProc) return { ok: false, error: 'A generation is already running.' };
  const cfg = loadConfig();
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outPath = path.join(OUT_DIR, `ideogram-${stamp}.png`);

  const args = [
    // low-memory wrapper: meta-device init + assign-load so the 2×9.3B DiTs and
    // the 8B text encoder never materialize as fp32 random inits (~60 GB peak)
    path.join(__dirname, 'python', 'run_lowmem.py'),
    path.join(REPO_DIR, 'run_inference.py'),
    '--prompt', params.prompt,
    '--output', outPath,
    '--width', String(params.width || 1024),
    '--height', String(params.height || 1024),
    '--seed', String(params.seed ?? 0),
    '--quantization', 'fp8', // nf4 is CUDA-only; fp8 runs on Apple Silicon
  ];
  if (params.preset) args.push('--sampler-preset', params.preset);
  if (cfg.ideogramKey) {
    args.push('--magic-prompt-key', cfg.ideogramKey);
  } else {
    // Without an Ideogram API key the prompt is fed to the model verbatim.
    // The model expects a structured JSON caption (magic prompt normally
    // produces it), so plain text trips the caption verifier — downgrade
    // that to a warning instead of aborting the generation.
    args.push('--no-magic-prompt', '--warn-on-caption-issues');
  }

  const env = { ...process.env, PATH: ENV_PATH, PYTHONUNBUFFERED: '1' };
  if (cfg.hfToken) { env.HF_TOKEN = cfg.hfToken; env.HUGGING_FACE_HUB_TOKEN = cfg.hfToken; }

  return new Promise((resolve) => {
    genProc = spawn(PYTHON, args, { env, cwd: REPO_DIR });
    let tail = '';
    const onData = (d) => {
      const text = d.toString();
      tail = (tail + text).slice(-4000);
      send('gen:log', text);
    };
    genProc.stdout.on('data', onData);
    genProc.stderr.on('data', onData);
    genProc.on('error', (err) => { genProc = null; resolve({ ok: false, error: String(err) }); });
    genProc.on('close', (code, signal) => {
      genProc = null;
      if (signal) resolve({ ok: false, cancelled: true });
      else if (code === 0 && fs.existsSync(outPath)) resolve({ ok: true, image: outPath });
      else resolve({ ok: false, error: `Inference exited with code ${code}.\n${tail.slice(-1500)}` });
    });
  });
});

ipcMain.handle('gen:cancel', () => {
  if (genProc) { genProc.kill('SIGTERM'); return true; }
  return false;
});

// ---------------------------------------------------------------------------
// History / misc
// ---------------------------------------------------------------------------
ipcMain.handle('history:list', () => {
  try {
    return fs.readdirSync(OUT_DIR)
      .filter((f) => f.endsWith('.png'))
      .map((f) => {
        const full = path.join(OUT_DIR, f);
        return { path: full, mtime: fs.statSync(full).mtimeMs };
      })
      .sort((a, b) => b.mtime - a.mtime)
      .slice(0, 50);
  } catch { return []; }
});

ipcMain.handle('shell:reveal', (_e, p) => shell.showItemInFolder(p));
ipcMain.handle('shell:open', (_e, url) => shell.openExternal(url));
ipcMain.handle('shell:openOutputs', () => shell.openPath(OUT_DIR));
