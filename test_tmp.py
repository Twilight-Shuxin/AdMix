import os
import subprocess

# ── LLM config ────────────────────────────────────────────────────────────────
llm_configs = {
    "llama7b": {"devices": "1"},
    "qwen14b": {"devices": "0,1"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_exp(cmd, data_path, subdir):
    full_cmd = f"{cmd} --dcgm_group 0 --rec-pw --path {data_path}/{subdir}"
    print(full_cmd)
    outfile = f"{data_path}/{subdir}/summary/out.out"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "a") as f:
        f.write(full_cmd + "\n")
        f.flush()
        subprocess.run(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

def gmanager(action, devices):
    subprocess.run(f"gmanager {action} {devices}", shell=True)

# ── Main ──────────────────────────────────────────────────────────────────────
os.environ["ADMIX_LLM_SERVING_PLATFORM"] = "vllm"
llm_model_tag_list = ["llama7b", "qwen14b"]

# ── Side-model config ──────────────────────────────────────────────────────────
models = {
    "resnet50":  {"mps_bs": 128, "time_bs": 4},
    "vgg19":     {"mps_bs": 128, "time_bs": 4},
    "unet3d":    {"mps_bs": 16,  "time_bs": 1},
    "deeplabv3": {"mps_bs": 32,  "time_bs": 2},
}

# ── Traces (each entry binds a trace to its MPS value) ───────────────────────
traces_map = {
    # "llama7b": [
        # {"trace": "BurstGPT_pattern", "mps": 20},
        # {"trace": "Azure1_pattern",    "mps": 20},
        # {"trace": "Azure2_pattern",    "mps": 20},
    # ],
    "llama7b": [
        # {"trace": "BurstGPT_npref_7b_d1_2", "mps": 15},
        # {"trace": "Azure_npref_7b_d1_1",    "mps": 10},
    ],
    "qwen14b": [
        {"trace": "BurstGPT_npref_14b_d2_1", "mps": 15},
        {"trace": "Azure_npref_14b_d2_1", "mps": 5}, 
    ]
}

for llm_model_tag in llm_model_tag_list:
    devices   = llm_configs[llm_model_tag]["devices"]
    data_path = f"data_final/{llm_model_tag}"
    traces    = traces_map[llm_model_tag]

    # # Utils run: LLM only
    # gmanager("stopmps", devices)
    # for tc in traces:
    #     run_exp(
    #         f"python run.py --model llm --llm_model {llm_model_tag}"
    #         f" --trace {tc['trace']} --devices {devices}",
    #         data_path, "utils",
    #     )

    # MPS run: LLM + side model
    gmanager("startmps", devices)
    for tc in traces:
        for m, cfg in models.items():
            run_exp(
                f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
                f" --scheduler constant[{cfg['mps_bs']}] --mps 100,{tc['mps']}"
                f" --trace {tc['trace']} --devices {devices}",
                data_path, "mps",
            )
    gmanager("stopmps", devices)

    # Context run: LLM + side model, modified vllm
    # gmanager("startmps", devices)
    # for tc in traces:
    #     for m in models:
    #         run_exp(
    #             f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
    #             f" --scheduler context"
    #             f" --trace {tc['trace']} --devices {devices} ",
    #             data_path, "context",
    #         )
    # gmanager("stopmps", devices)

    # # Batch run: LLM + side model, modified vllm
    # for tc in traces:
    #     for m in models:
    #         gmanager("startmps", devices)
    #         run_exp(
    #             f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
    #             f" --scheduler batch"
    #             f" --trace {tc['trace']} --devices {devices}",
    #             data_path, "batch",
    #         )

    # # Time run: LLM + side model, no MPS
    # gmanager("stopmps", devices)
    # for tc in traces:
    #     for m, cfg in models.items():
    #         run_exp(
    #             f"python run.py --model llm,{m} --llm_model {llm_model_tag}"
    #             f" --scheduler constant[{cfg['time_bs']}]"
    #             f" --trace {tc['trace']} --devices {devices}",
    #             data_path, "time",
    #         )
