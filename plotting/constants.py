import os

foo = os.environ.get('GVLLM_PLOTTING_BIN_SIZE')
if(foo is not None):
    PLOTTING_BIN_SIZE = float(foo)
else:
    PLOTTING_BIN_SIZE = 0.5