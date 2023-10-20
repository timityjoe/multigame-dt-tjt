#!/bin/bash

# See https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html
echo "Setting up Multi-Game Decision Transformer Environment..."
source activate base	
conda deactivate
conda activate conda39-mgdt
echo "$PYTHON_PATH"
