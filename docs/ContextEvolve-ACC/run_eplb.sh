#!/bin/bash
# 运行 EPLB 场景的脚本

# 设置 API key（请替换为你的实际 API key）
export GEMINI_API_KEY="your_gemini_api_key_here"

# 或者如果使用 OpenAI 兼容的 API
# export OPENAI_API_KEY="your_api_key_here"
# export OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"

# 切换到工作目录
cd /home/dataset-local/shy/lab/icml26/ADRS

# 运行 OpenEvolve
python openevolve-run.py \
  openevolve/examples/ADRS/eplb/initial_program.py \
  openevolve/examples/ADRS/eplb/evaluator.py \
  --config openevolve/examples/ADRS/eplb/config.yaml \
  --output openevolve/examples/ADRS/eplb/output \
  --iterations 100 \
  --log-level INFO

