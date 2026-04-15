# MPSTest

## Environment notes

The uv environment can be built with the following commands (but with lock file can be reproduced directly using `uv sync`, only as a note for myself in case necessary to build environment from scratch again).

### Trace & Dataset

The prompt generation utilizes ShareGPT prompt set available at: 
```
wget https://raw.githubusercontent.com/HPMLL/BurstGPT/main/example/preprocess_data/shareGPT.jso
```
Please download this into folder `./dataset/`.

The BurstGPT and Azure traces download & setup scripts are available at `download_traces.sh` under `./trace/`

#### VLLM

The environment needed for using vllm as backend:

```bash
uv add torch==2.8.0
uv add numpy==2.2.6
uv add vllm==0.10.2
uv add scikit-learn
uv add matplotlib
```

Use a slienced version of vllm: Reference my own comment @ https://github.com/vllm-project/vllm/issues/5338

Before running Admix: 
```
export ADMIX_LLM_SERVING_PLATFORM=vllm
```

#### Sglang

The environment Sglang is to be built under `plugins/sglang_env`, and can be built as follows:

```bash
uv add sglang
uv pip install sgl-kernel --force-reinstall --no-deps
```

To silence the logs of cancelling requests: modify `sglang/srt/entrypoints/http_server.py` line 544-546 & 563-565: add following code to exception catch.
```
if "disconnected from the client side" not in str(e):
    logger.error(f"[http_server] Error: {e}")
```

Before running Admix:
```
export ADMIX_LLM_SERVING_PLATFORM=sglang
```
Note that running Admix this way Admix will still be using the default environment built for vllm, only the sglang server is referencing the specific environment.

## Usage

```
usage: run.py [-h] --model MODEL [--mps MPS] [--batch_size BATCH_SIZE] [--start START] [--end END]
              [--dcgm_group DCGM_GROUP] [--devices DEVICES] [--path PATH] [--profile_path PROFILE_PATH]
              [--trace TRACE] [--seq_len SEQ_LEN] [--tag TAG] [--test_time TEST_TIME] [--mode MODE]
              [--scheduler SCHEDULER] [--rec-pw]
```

Many arguments accept comma-separated values (one entry per model when running multiple models).

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model MODEL` | *(required)* | Comma-separated list of model names to run. |
| `--mps MPS` | `""` | Comma-separated list of CUDA MPS active thread percentages per model. If fewer values than models are given, the first value is used for the rest. If not provided, all set to 100. |
| `--batch_size BATCH_SIZE` | `""` | Comma-separated list of batch sizes per model (defaults to 1 per model, ignored for llm). |
| `--start START` | `""` | Comma-separated list of start operator indices per model (for partial/intermediate execution). |
| `--end END` | `""` | Comma-separated list of end operator indices per model. |
| `--dcgm_group DCGM_GROUP` | `-1` | Comma-separated DCGM group IDs for GPU metric monitoring via `dcgm`. |
| `--devices DEVICES` | `0` | Comma-separated GPU device IDs to use (e.g. `0,1`). |
| `--path PATH` | `data` | Directory (relative to `MPSTest/`) to save test log output. |
| `--profile_path PROFILE_PATH` | `profile_data` | Directory (relative to `MPSTest/`) to save data collected in `PROFILE` mode. |
| `--trace TRACE` | `""` | Comma-separated list of trace file paths (or trace config id for llm) per model. |
| `--seq_len SEQ_LEN` | `""` | Sequence length for LLM workers, as `prompt_len,output_len`. |
| `--tag TAG` | `None` | Tag string attached to result records for identification. |
| `--test_time TEST_TIME` | `100` | The total test time (seconds). |
| `--mode MODE` | `TEST` | Runtime mode. One of: `TEST` (default benchmark), `PROFILE` (single-model profiling), `SERVE` (LLM serving with scheduler). |
| `--scheduler SCHEDULER` | `None` | Scheduler to use in `SERVE` mode (if specified, mode is set to SERVE by default). |
| `--rec-pw` | *(flag)* | If set, record GPU power/energy usage during the run. |

### Examples:
```
uv run run.py --model resnet50 --batch_size 1 --devices 0 --test_time 50
uv run run.py --model resnet50 --batch_size 1 --devices 0 --mps 50 --test_time 50
uv run run.py --model resnet50 --batch_size 1 --devices 0 --test_time 50 --mode PROFILE
uv run run.py --model llm --devices 0 --trace test
uv run run.py --model llm,resnet50 --devices 0 --trace test --scheduler batch
```