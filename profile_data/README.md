# Profile Data

This folder holds the profiled data of the opportunistic workloads. 
The profiled data are stored in json format as shown in the following example.

This is the profile of the model GAN, where with batch size 1 the 99 percentile latency is 0.008s and the average energy per inference = 0.5J.

```
{
    "model_name": "GAN",
    "1": {
        "latency": 0.0080, 
        "avg_energy": 0.5
    }
}
```

Currently, to reduce the time spent on profiling, the average energy per inference is only profiled for the configured maximum batch size of each model.