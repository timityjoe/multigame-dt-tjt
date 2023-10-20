# Implementation of Multi-Game Decision Transformers in PyTorch

## Quickstart
```bash
conda create --name conda39-mgdt python=3.9
conda activate conda39-mgdt
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113^C
pip install -r requirements.txt

pip install "gym[atari, accept-rom-license]"

python3 scripts/download_weights.py
python3 run_atari.py
```

## Baselines

> [logs](workdir/)

| model | params | task     | this repo | orig. |
| ----- | ------ | -------- | --------- | ----- |
| mgdt  | 200M   | Breakout | 298.8     | 290.6 |

## References:

- [1] Original code in Jax: https://github.com/google-research/google-research/tree/master/multi_game_dt
- [2] Paper: https://arxiv.org/abs/2205.15241
