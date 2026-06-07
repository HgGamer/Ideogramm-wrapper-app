"""Low-memory launcher for ideogram4's run_inference.py.

Upstream ``Ideogram4Pipeline.from_pretrained`` materializes every component
with a full random init on CPU before the quantized checkpoint is loaded:
~37 GB of fp32 for EACH 9.3B transformer (there are two) plus ~32 GB for the
Qwen3-VL-8B text encoder. Peak demand exceeds 60 GB and thrashes any machine
below ~64 GB of unified memory.

This wrapper monkeypatches the fp8 builders to initialize parameters on the
meta device (``accelerate.init_empty_weights``) and load the checkpoint with
``assign=True``, so the only real allocations are the fp8 tensors themselves
(~9 GB per transformer, ~8 GB for the text encoder). Computed non-persistent
buffers (rotary ``inv_freq`` / caches) stay real via ``include_buffers=False``.

It also frees the text encoder right after the prompt is encoded — it is not
needed during the denoising loop, and this process generates exactly one
image and exits.

Finally, it fixes Fp8Linear on Apple Silicon: the MPS backend cannot cast
float8_e4m3fn to bf16 (``weight.to(x.dtype)`` raises TypeError), so on MPS we
dequantize through a 256-entry lookup table instead — view the fp8 bits as
uint8 and gather. The fp8 weights stay resident as 1 byte/param.

Usage:  python run_lowmem.py /path/to/run_inference.py [run_inference args...]
"""

from __future__ import annotations

import gc
import runpy
import sys

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

from ideogram4 import pipeline_ideogram4 as pl
from ideogram4.modeling_ideogram4 import Ideogram4Transformer
from ideogram4 import quantized_loading as ql
from ideogram4.quantized_loading import (
  is_fp8_state_dict,
  load_fp8_state_dict,
  swap_linears_to_fp8,
)

# ---------------------------------------------------------------------------
# Fp8Linear on MPS: dequantize via lookup table (MPS can't cast float8 → bf16)
# ---------------------------------------------------------------------------
_LUT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _fp8_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  key = (device, dtype)
  lut = _LUT_CACHE.get(key)
  if lut is None:
    # Decode all 256 e4m3fn bit patterns on CPU (which does support the cast).
    bits = torch.arange(256, dtype=torch.uint8)
    lut = bits.view(torch.float8_e4m3fn).to(torch.float32).nan_to_num(0.0)
    lut = lut.to(device=device, dtype=dtype)
    _LUT_CACHE[key] = lut
  return lut


_orig_fp8_forward = ql.Fp8Linear.forward


def _fp8_forward_mps_safe(self, x: torch.Tensor) -> torch.Tensor:
  if self.weight.device.type != "mps":
    return _orig_fp8_forward(self, x)
  lut = _fp8_lut(self.weight.device, x.dtype)
  w = lut[self.weight.view(torch.uint8).long()]
  w = w * self.weight_scale.to(x.dtype).unsqueeze(1)
  bias = self.bias.to(x.dtype) if self.bias is not None else None
  return torch.nn.functional.linear(x, w, bias)

# ---------------------------------------------------------------------------
# LogitNormalSchedule on MPS: the schedule math runs in float64, which the MPS
# backend doesn't support. Hop to CPU for the (tiny, scalar-sized) computation
# and return the float32 result on the original device — numerically identical.
# ---------------------------------------------------------------------------
from ideogram4 import scheduler as sched

_orig_schedule_call = sched.LogitNormalSchedule.__call__


def _schedule_call_mps_safe(self, t: torch.Tensor) -> torch.Tensor:
  if t.device.type != "mps":
    return _orig_schedule_call(self, t)
  return _orig_schedule_call(self, t.cpu()).to(t.device)


_orig_build_transformer = pl._build_transformer


def _build_transformer_lowmem(transformer_config, state_dict, device, dtype):
  """Meta-init + assign-load for fp8; defer to upstream for other formats."""
  if not is_fp8_state_dict(state_dict):
    return _orig_build_transformer(transformer_config, state_dict, device, dtype)
  with init_empty_weights(include_buffers=False):
    model = Ideogram4Transformer(transformer_config)
  swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
  # strict=True: every meta param must be filled from the checkpoint.
  load_fp8_state_dict(model, state_dict, device=device, dtype=dtype, assign=True)
  model.eval()
  return model


def _load_fp8_text_encoder_lowmem(repo_id, device, dtype, *, text_encoder_subfolder):
  """Same as upstream _load_fp8_text_encoder, but params init on meta."""
  config = AutoConfig.from_pretrained(
    repo_id, subfolder=text_encoder_subfolder, trust_remote_code=True
  )
  with init_empty_weights(include_buffers=False):
    model = AutoModel.from_config(config, trust_remote_code=True)
  state_dict = pl._load_subfolder_state_dict(repo_id, text_encoder_subfolder, "model")
  swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
  # strict=False as upstream: tied weights surface as benign missing keys.
  load_fp8_state_dict(
    model, state_dict, device=device, dtype=dtype, assign=True, strict=False
  )
  model.eval()
  return model


_orig_encode_text = pl.Ideogram4Pipeline._encode_text


def _encode_text_and_release(self, *args, **kwargs):
  """Encode the prompt, then drop the 8B text encoder (~8 GB).

  Safe only because this process generates a single image and exits — a
  second pipe() call would find text_encoder gone.
  """
  out = _orig_encode_text(self, *args, **kwargs)
  self.text_encoder = None
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()
  return out


ql.Fp8Linear.forward = _fp8_forward_mps_safe
sched.LogitNormalSchedule.__call__ = _schedule_call_mps_safe
pl._build_transformer = _build_transformer_lowmem
pl._load_fp8_text_encoder = _load_fp8_text_encoder_lowmem
pl.Ideogram4Pipeline._encode_text = _encode_text_and_release


def main() -> None:
  if len(sys.argv) < 2:
    print("usage: run_lowmem.py /path/to/run_inference.py [args...]", file=sys.stderr)
    sys.exit(2)
  sys.argv = sys.argv[1:]
  runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
  main()
