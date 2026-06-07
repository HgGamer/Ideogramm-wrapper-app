const { contextBridge, ipcRenderer } = require('electron');

const on = (channel) => (cb) => {
  const handler = (_e, payload) => cb(payload);
  ipcRenderer.on(channel, handler);
  return () => ipcRenderer.removeListener(channel, handler);
};

contextBridge.exposeInMainWorld('studio', {
  setupState: () => ipcRenderer.invoke('setup:state'),
  runSetup: () => ipcRenderer.invoke('setup:run'),
  setKeys: (keys) => ipcRenderer.invoke('config:setKeys', keys),
  presets: () => ipcRenderer.invoke('gen:presets'),
  generate: (params) => ipcRenderer.invoke('gen:start', params),
  cancel: () => ipcRenderer.invoke('gen:cancel'),
  history: () => ipcRenderer.invoke('history:list'),
  reveal: (p) => ipcRenderer.invoke('shell:reveal', p),
  openExternal: (url) => ipcRenderer.invoke('shell:open', url),
  openOutputs: () => ipcRenderer.invoke('shell:openOutputs'),

  onSetupLog: on('setup:log'),
  onSetupStep: on('setup:step'),
  onSetupDone: on('setup:done'),
  onSetupError: on('setup:error'),
  onGenLog: on('gen:log'),
});
