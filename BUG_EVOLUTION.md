# Bug Evolution — Key Learnings

A chronological record of significant bugs encountered during development and the lessons learned.

---

## 1. TorchScript vs. Pickle (.pt)

**Error**: `torch::jit::load()` fails or produces wrong behavior when loading a `torch.save()` model.

**Cause**: `torch.save()` serializes the Python object graph as a pickle. LibTorch C++ expects TorchScript, which is a different IR.

**Fix**:
```python
# Wrong
torch.save(model.state_dict(), "model.pt")

# Correct
traced = torch.jit.trace(wrapper, dummy_input, check_trace=False)
traced.save("model.pt")
```

**Learning**: C++ LibTorch requires TorchScript export via `torch.jit.trace()` or `torch.jit.script()`, not `torch.save()`.

---

## 2. aten::empty_strided CUDA Backend Not Available

**Error**:
```
Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend.
'aten::empty_strided' is only available for these backends: [CPU, Meta, ...]
```

**Cause**: LibTorch build is CPU-only. CUDA kernels for basic ops like `empty_strided` are missing. Often due to:
- Wrong LibTorch zip (CPU instead of cu126)
- Incomplete or custom LibTorch build

**Fix**: Add a CPU inference fallback in `InferenceEngine`:
1. Try `model.to(torch::kCUDA)`
2. On exception: set `device = torch::kCPU`, allocate host buffers
3. In `forward()`: if CPU, copy `d_input` D2H, run on CPU, copy output H2D to `d_output`

**Learning**: Always handle CPU-only LibTorch. Preprocessing/postprocessing can stay on GPU; only inference moves to CPU when needed.

---

## 3. Trace on CPU to Avoid CUDA Device References

**Error**: Same `aten::empty_strided` error when loading a model that was traced on GPU.

**Cause**: If traced with CUDA tensors, the saved TorchScript may embed CUDA device references. When loaded in CPU-only LibTorch, moving such a model to CPU can still trigger CUDA ops internally.

**Fix**:
```python
raw_model = model.model.cpu()
dummy_input = torch.rand(1, 3, 640, 640, device="cpu")
traced = torch.jit.trace(wrapper, dummy_input, check_trace=False)
traced.save(output_path)
```

**Learning**: For maximum portability (especially with CPU LibTorch), trace models on CPU.

---

## 4. Graphs Differed Across Invocations

**Error**:
```
Tracing failed sanity checks!
ERROR: Graphs differed across invocations!
```

**Cause**: YOLOv5 has paths that can vary between runs (e.g., internal conditionals, batchnorm behavior). The trace sanity check re-runs the model and compares graphs; they differ.

**Fix**:
```python
traced = torch.jit.trace(wrapper, dummy_input, check_trace=False)
```

**Learning**: For models with non-deterministic or slightly varying graph structure, `check_trace=False` can be acceptable if the traced model produces correct results in practice.

---

## 5. cudaMemcpy Illegal Memory Access (D2H Input)

**Error**:
```
CUDA Error: cudaMemcpy D2H input failed: an illegal memory access was encountered
```

**Cause**: The preprocessing kernel was passed `image->data` (host pointer) instead of `image->d_data` (device pointer). The kernel tried to read from host memory as if it were device memory.

**Fix**: After `copyImageToDevice()`, use `image->d_data` in the kernel:
```cpp
unsigned char* d_src = image->d_data;
letterboxResizeKernel<<<...>>>(d_src, d_output, ...);
```

**Learning**: In CUDA, clearly distinguish host (`data`) vs device (`d_data`) pointers. Never pass a host pointer to a kernel that expects device memory.

---

## 6. Image Struct: data vs. d_data

**Observation**: `Image` has both `data` (host) and `d_data` (device). `copyImageToDevice()` fills `d_data` but does not change `data`. Code that assumed `data` pointed to device memory after copy was wrong.

**Learning**: Document and enforce the invariant: `data` = host, `d_data` = device. When device copy exists, kernels must use `d_data`.

---

## Summary Table

| # | Bug / Issue | Root Cause | Fix |
|---|-------------|------------|-----|
| 1 | Wrong model format | Pickle vs TorchScript | Use `torch.jit.trace` + `.save()` |
| 2 | aten::empty_strided CUDA | CPU-only LibTorch | CPU inference fallback |
| 3 | CUDA refs in traced model | Traced on GPU | Trace on CPU |
| 4 | Trace sanity check fails | Varying graph across runs | `check_trace=False` |
| 5 | Illegal memory access in cudaMemcpy | Host ptr in kernel | Use `d_data` in preprocessing kernel |
| 6 | data vs d_data confusion | Unclear ownership | Document: data=host, d_data=device |

---

## Prevention Checklist

- [ ] Export model with `torch.jit.trace` to TorchScript; never use `torch.save` for C++ deployment
- [ ] Trace on CPU for LibTorch portability
- [ ] Implement CPU inference fallback when `model.to(CUDA)` fails
- [ ] Use `check_trace=False` for YOLOv5-style models if sanity check fails
- [ ] Pass only device pointers (`d_*`) to CUDA kernels
- [ ] After `copyImageToDevice()`, use `image->d_data` in kernels, not `image->data`
