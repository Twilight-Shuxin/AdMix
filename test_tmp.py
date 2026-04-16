import os
import subprocess

# ── LLM config ────────────────────────────────────────────────────────────────
llm_configs = {
    "llama7b": {"devices": "1"},
    "qwen14b": {"devices": "0,1"},
}

llm_model_tag = "qwen14b"
devices       = llm_configs[llm_model_tag]["devices"]
data_path     = f"data_staged/{llm_model_tag}"

# ── Side-model config ──────────────────────────────────────────────────────────
models = {
    "resnet50":  {"mps_bs": 128, "time_bs": 4},
    "vgg19":     {"mps_bs": 128, "time_bs": 4},
    "unet3d":    {"mps_bs": 16,  "time_bs": 1},
    "deeplabv3": {"mps_bs": 32,  "time_bs": 2},
}

# ── Traces (each entry binds a trace to its MPS value) ───────────────────────
traces = [
    {"trace": "BurstGPT_npref_7b_d2_1", "mps": 20},
    {"trace": "Azure_npref_7b_d2_1", "mps": 10}
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_cmd(cmd, outfile=None):
    print(cmd)
    if outfile:
        with open(outfile, "a") as f:
            f.write(cmd + "\n")
            f.flush()
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, shell=True)

def gmanager(action):
    subprocess.run(f"gmanager {action} {devices}", shell=True)

# ── Main ──────────────────────────────────────────────────────────────────────
os.environ["ADMIX_LLM_SERVING_PLATFORM"] = "vllm"

# # Utils run: LLM only
# gmanager("stopmps")
# for tc in traces:
#     if(tc['trace'].startswith("BurstGPT")):
#         run_cmd(
#             f"python run.py --model llm --llm_model {llm_model_tag}"
#             f" --dcgm_group 0 --trace {tc['trace']} --devices {devices}"
#             f" --rec-pw --path {data_path}/utils",
#             "out.out",
#         )

# # MPS run: LLM + side model
# for tc in traces:
#     for m, cfg in models.items():
#         gmanager("startmps")
#         run_cmd(
#             f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
#             f" --scheduler constant[{cfg['mps_bs']}] --mps 100,{tc['mps']}"
#             f" --dcgm_group 0 --trace {tc['trace']} --devices {devices}"
#             f" --rec-pw --path {data_path}/mps",
#             "out_mps.out",
#         )

# Context run: LLM + side model, modified vllm
for tc in traces:
    for m in models:
        gmanager("startmps")
        run_cmd(
            f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
            f" --scheduler context"
            f" --dcgm_group 0 --trace {tc['trace']} --devices {devices}"
            f" --rec-pw --path {data_path}/context",
            "out_context.out",
        )

# Batch run: LLM + side model, modified vllm
for tc in traces:
    for m in models:
        gmanager("startmps")
        run_cmd(
            f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
            f" --scheduler batch"
            f" --dcgm_group 0 --trace {tc['trace']} --devices {devices}"
            f" --rec-pw --path {data_path}/batch",
            "out_batch.out",
        )

# Time run: LLM + side model, no MPS
gmanager("stopmps")
for tc in traces:
    for m, cfg in models.items():
        run_cmd(
            f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
            f" --scheduler constant[{cfg['time_bs']}]"
            f" --dcgm_group 0 --trace {tc['trace']} --devices {devices}"
            f" --rec-pw --path {data_path}/time",
            "out_time.out",
        )
