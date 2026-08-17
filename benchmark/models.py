"""Model registry — all models to benchmark with their OpenRouter pricing."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    model_id: str
    cost_per_1k_input: float   # USD per 1,000 input tokens
    cost_per_1k_output: float  # USD per 1,000 output tokens
    supports_vision: bool = False


MODELS = [
    ModelConfig(
        name="Claude Sonnet 4.5",
        model_id="anthropic/claude-sonnet-4-5",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_vision=True,
    ),
    ModelConfig(
        name="GPT-4o",
        model_id="openai/gpt-4o",
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        supports_vision=True,
    ),
    ModelConfig(
        name="GPT-4o Mini",
        model_id="openai/gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        supports_vision=True,
    ),
    ModelConfig(
        name="DeepSeek Chat",
        model_id="deepseek/deepseek-chat",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        supports_vision=False,
    ),
    ModelConfig(
        name="Llama 3.3 70B",
        model_id="meta-llama/llama-3.3-70b-instruct",
        cost_per_1k_input=0.00012,
        cost_per_1k_output=0.0003,
        supports_vision=False,
    ),
    ModelConfig(
        name="Mistral Large",
        model_id="mistralai/mistral-large",
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
        supports_vision=False,
    ),
    ModelConfig(
        name="Amazon Nova Lite",
        model_id="amazon/nova-lite-v1",
        cost_per_1k_input=0.00006,
        cost_per_1k_output=0.00024,
        supports_vision=True,
    ),
    ModelConfig(
        name="Amazon Nova Pro",
        model_id="amazon/nova-pro-v1",
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.0032,
        supports_vision=True,
    ),
    # --- Gemini models ---
    ModelConfig(
        name="Gemini 2.5 Flash",
        model_id="google/gemini-2.5-flash",
        cost_per_1k_input=0.00030,
        cost_per_1k_output=0.00250,
        supports_vision=True,
    ),
    ModelConfig(
        name="Gemini 2.5 Flash Lite",
        model_id="google/gemini-2.5-flash-lite",
        cost_per_1k_input=0.00010,
        cost_per_1k_output=0.00040,
        supports_vision=True,
    ),
    ModelConfig(
        name="Gemini 2.5 Pro",
        model_id="google/gemini-2.5-pro",
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.01000,
        supports_vision=True,
    ),
    # --- Chinese models ---
    ModelConfig(
        name="Qwen 2.5 VL 72B",
        model_id="qwen/qwen2.5-vl-72b-instruct",
        cost_per_1k_input=0.0004,
        cost_per_1k_output=0.0004,
        supports_vision=True,
    ),
    ModelConfig(
        name="Qwen 3 235B",
        model_id="qwen/qwen3-235b-a22b",
        cost_per_1k_input=0.00022,
        cost_per_1k_output=0.00088,
        supports_vision=False,
    ),
    ModelConfig(
        name="Qwen 3 30B",
        model_id="qwen/qwen3-30b-a3b",
        cost_per_1k_input=0.00010,
        cost_per_1k_output=0.00030,
        supports_vision=False,
    ),
    ModelConfig(
        name="DeepSeek R1",
        model_id="deepseek/deepseek-r1",
        cost_per_1k_input=0.00055,
        cost_per_1k_output=0.00219,
        supports_vision=False,
    ),
    ModelConfig(
        name="DeepSeek V3",
        model_id="deepseek/deepseek-chat-v3-0324",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        supports_vision=False,
    ),
    ModelConfig(
        name="Kimi K2",
        model_id="moonshotai/kimi-k2",
        cost_per_1k_input=0.00060,
        cost_per_1k_output=0.00250,
        supports_vision=False,
    ),
    ModelConfig(
        name="MiniMax",
        model_id="minimax/minimax-01",
        cost_per_1k_input=0.00040,
        cost_per_1k_output=0.00040,
        supports_vision=False,
    ),
    ModelConfig(
        name="Mistral Nemo",
        model_id="mistralai/mistral-nemo",
        cost_per_1k_input=0.00013,
        cost_per_1k_output=0.00013,
        supports_vision=False,
    ),
    # --- New cheap models ---
    ModelConfig(
        name="Gemma 3 27B",
        model_id="google/gemma-3-27b-it",
        cost_per_1k_input=0.00008,
        cost_per_1k_output=0.00045,
        supports_vision=True,
    ),
    ModelConfig(
        name="Gemma 3 12B",
        model_id="google/gemma-3-12b-it",
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00015,
        supports_vision=True,
    ),
    ModelConfig(
        name="Gemma 3n E4B",
        model_id="google/gemma-3n-e4b-it",
        cost_per_1k_input=0.00006,
        cost_per_1k_output=0.00012,
        supports_vision=False,
    ),
    ModelConfig(
        name="Phi-4",
        model_id="microsoft/phi-4",
        cost_per_1k_input=0.00007,
        cost_per_1k_output=0.00014,
        supports_vision=False,
    ),
    ModelConfig(
        name="Llama 4 Scout",
        model_id="meta-llama/llama-4-scout",
        cost_per_1k_input=0.00010,
        cost_per_1k_output=0.00030,
        supports_vision=True,
    ),
    ModelConfig(
        name="Llama 4 Maverick",
        model_id="meta-llama/llama-4-maverick",
        cost_per_1k_input=0.00020,
        cost_per_1k_output=0.00080,
        supports_vision=True,
    ),
    ModelConfig(
        name="Cohere Command R7B",
        model_id="cohere/command-r7b-12-2024",
        cost_per_1k_input=0.00004,
        cost_per_1k_output=0.00015,
        supports_vision=False,
    ),
    ModelConfig(
        name="GPT-4.1 Mini",
        model_id="openai/gpt-4.1-mini",
        cost_per_1k_input=0.00040,
        cost_per_1k_output=0.00160,
        supports_vision=True,
    ),
    ModelConfig(
        name="GPT-4.1 Nano",
        model_id="openai/gpt-4.1-nano",
        cost_per_1k_input=0.00010,
        cost_per_1k_output=0.00040,
        supports_vision=True,
    ),
    ModelConfig(
        name="o4-mini",
        model_id="openai/o4-mini",
        cost_per_1k_input=0.00110,
        cost_per_1k_output=0.00440,
        supports_vision=True,
    ),
    # --- DeepSeek V4 ---
    ModelConfig(
        name="DeepSeek V4 Flash",
        model_id="deepseek/deepseek-v4-flash",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        supports_vision=False,
    ),
    ModelConfig(
        name="DeepSeek V4 Pro",
        model_id="deepseek/deepseek-v4-pro",
        cost_per_1k_input=0.00044,
        cost_per_1k_output=0.00088,
        supports_vision=False,
    ),
    # --- Tencent Hunyuan ---
    ModelConfig(
        name="Hunyuan A13B",
        model_id="tencent/hunyuan-a13b-instruct",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00057,
        supports_vision=False,
    ),
    ModelConfig(
        name="Hunyuan HY3",
        model_id="tencent/hy3",
        cost_per_1k_input=0.00013,
        cost_per_1k_output=0.00053,
        supports_vision=False,
    ),
]
