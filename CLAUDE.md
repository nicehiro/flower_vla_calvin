# Project Notes

## Known Issues & Fixes

### Florence-2 `_supports_sdpa` AttributeError

**Error:**
```
AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'
```

**Cause:** Transformers 4.54.0+ checks `_supports_sdpa` during model init, but Florence-2's cached custom code defines it as a property that references `self.language_model` (which doesn't exist yet during init).

**Fix:** Patch the cached HuggingFace model file:

```bash
# File location:
/data/fywang/hf_cache/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large/21a599d414c4d928c9032694c424fb94458e3594/modeling_florence2.py
```

Find the `Florence2PreTrainedModel` class (~line 2328) and replace:

```python
# Before (broken):
@property
def _supports_flash_attn_2(self):
    return self.language_model._supports_flash_attn_2

@property
def _supports_sdpa(self):
    return self.language_model._supports_sdpa
```

With:

```python
# After (fixed):
_supports_flash_attn_2 = True
_supports_sdpa = True
```

**Note:** This patch is lost if you clear the HF cache or re-download the model. Re-apply if needed.

**Reference:** https://github.com/huggingface/transformers/issues/39974

---

### calvin_env Editable Install Required

**Error:**
```
[Errno 2] No such file or directory: PosixPath('/root/flower_vla_calvin/.venv/lib/python3.9/site-packages/egl_check')
```

**Cause:** `calvin_env` uses relative path resolution (`Path(__file__).absolute().parents[2]`) to find the `egl_check` directory. When installed as a regular package in site-packages, the path resolves incorrectly.

**Fix:** Install `calvin_env` in editable mode:
```bash
cd /root/flower_vla_calvin/calvin_env && uv pip install -e .
```

---

### EGL Build Dependencies

**Error:**
```
fatal error: X11/Xlib.h: No such file or directory
```

**Cause:** Missing X11 and EGL development libraries needed to build `egl_check/EGL_options.o`.

**Fix:**
```bash
apt-get install -y libx11-dev libegl1-mesa-dev libgles2-mesa-dev libosmesa6-dev
cd /root/flower_vla_calvin/calvin_env/egl_check && bash build.sh
```

---

### calvin_env `__file__` is None

**Error:**
```
expected str, bytes or os.PathLike object, not NoneType
```

**Cause:** `calvin_env.__file__` returns `None` with certain editable install configurations. Line 72 in `play_table_env.py` tries to get git commit hash using this path.

**Fix:** Already patched in `calvin_env/calvin_env/envs/play_table_env.py`:
```python
if calvin_env.__file__ is not None:
    log.info(f"Using calvin_env with commit {get_git_commit_hash(Path(calvin_env.__file__))}.")
```

---

### Language Embeddings Setup for calvin_debug_dataset

**Error:**
```
FileNotFoundError: '.../validation/lang_clip_resnet50/embeddings.npy'
```

**Cause:** The `lang_clip_resnet50` embeddings are not included in the debug dataset download.

**Fix:** Copy embeddings from `task_ABCD_D` dataset, but replace `auto_lang_ann.npy` with the correct one:

```bash
# Copy embeddings (same 34 tasks, same annotations)
cp -r /data/fywang/Calvin/task_ABCD_D/validation/lang_clip_resnet50 /data/fywang/Calvin/calvin_debug_dataset/validation/
cp -r /data/fywang/Calvin/task_ABCD_D/training/lang_clip_resnet50 /data/fywang/Calvin/calvin_debug_dataset/training/

# IMPORTANT: Replace auto_lang_ann.npy with correct episode indices
cp /data/fywang/Calvin/calvin_debug_dataset/validation/auto_lang_ann.npy /data/fywang/Calvin/calvin_debug_dataset/validation/lang_clip_resnet50/
cp /data/fywang/Calvin/calvin_debug_dataset/training/auto_lang_ann.npy /data/fywang/Calvin/calvin_debug_dataset/training/lang_clip_resnet50/
```

**Why:** The `embeddings.npy` contains task embeddings (task name → embedding vector) which are dataset-independent. However, `auto_lang_ann.npy` contains episode indices that differ between datasets:
- `task_ABCD_D`: episodes start at index ~40
- `calvin_debug_dataset`: episodes start at index ~553567

**Also update config** (`conf/config_calvin.yaml`):
```yaml
goal_dim: 1024  # lang_clip_resnet50 uses 1024-dim embeddings, not 512
```
