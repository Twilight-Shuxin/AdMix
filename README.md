# MPSTest

## Environment notes
The uv environment is built with the following commands (but with lock file can be reproduced directly using `uv sync`, only as a note for myself in case necessary to build environment from scratch again).

```bash
uv add torch==2.8.0
uv add numpy==2.2.6
uv add vllm==0.10.2
uv add scikit-learn
uv add matplotlib
```

Use a slienced version of vllm: Reference my own comment @ https://github.com/vllm-project/vllm/issues/5338

#### VLLM

```
export ADMIX_LLM_SERVING_PLATFORM=vllm
```

#### Sglang

```
export ADMIX_LLM_SERVING_PLATFORM=sglang
```