# FastVideo masked attention adapter validation

The [GitHub Actions run](https://github.com/bingdongni/fastvideo-gpu-validation-20261108-2c92df8c0d/actions/runs/42464269396) validates FastVideo commit `b03d2c63ef8414ee71839e88747546593d911c29` against upstream baseline `ce970eb715215c79a169658d7aa6c8a96e64891e`. The baseline produced four failing and two passing cases; the patched code passed all six. The [job log](gpu-validation.log) records the checks and GPU details. The repository hooks also passed on the changed files in the same run.

The one-job runner used Windows, an NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.6.0+cu124 and CUDA 12.4. The Python adapter is the FastVideo implementation, but this check substitutes PyTorch CUDA scaled-dot-product attention for the FlashAttention no-padding functions. It therefore tests GPU tensor and adapter semantics, not a native FlashAttention kernel or end-to-end video generation.
