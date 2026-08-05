target_dir="/slot1/teleems_models_datasets/models/bioner/Meta-Llama-3-8B"
port=16161

python -m vllm.entrypoints.openai.api_server \
  --model "$target_dir" \
  --tokenizer "$target_dir" \
  --port "$port" \
  --served-model-name "llama3-8b" \
  # --gpu-memory-utilization 0.9 \
  # --max-model-len 1024 \