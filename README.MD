# LLM Guardrails Assessment

**Based on the code given by Dr. Ricardo Cardoso Pereira at https://ricardodcpereira.com/**

This project evaluates the effectiveness of different safety guardrails in Large Language Models (LLMs) when facing adversarial inputs. It tests lightweight, inference level safeguards, such as **Supervisory Agents** and **Self-Critique Loops**, across multiple models (Mistral 7B, LLaMA 3.3 70B, and DeepSeek R1) using the OpenRouter API.

## Project Structure

The experiments are divided into five Python scripts, representing different safety configurations:

* **`1_Baseline.py`**: The control group. Tests models with a standard, safe system prompt ("Keep your answers concise...").
* **`2_Mean_Baseline.py` (Unsafe Baseline)**: The adversarial control. Forces the model into an "unsafe assistant" persona to establish a baseline for failure.
* **`3_Reverse_Injection.py`**: **Input Guardrail.** Pre-processes the user prompt by injecting a "safety prefix" instruction before the model sees the harmful query.
* **`4_Supervisory_Agent.py`**: **Constitutional AI / Multi-Agent.** Uses a secondary "Supervisor" agent to review the main agent's output. If the output is harmful, the Supervisor requests a revision.
* **`5_Self_Critique_Loop.py`**: **Self-Correction.** The agent generates a response, then is prompted to critique its own output for hallucinations or illegality and rewrite it if necessary.

## Prerequisites

* Python 3.10 or higher.
* An active **OpenRouter API Key**.

## Installation

1.  **Clone or download** this repository.
2.  **Install the dependencies** using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

You must set your OpenRouter API key as an environment variable before running the scripts.

**Linux / macOS:**
```bash
export OPENROUTER_API_KEY="sk-or-your-key-here"
