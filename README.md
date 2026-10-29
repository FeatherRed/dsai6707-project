# dsai6707-project

## 1. Environment Setup

The project requires **Python 3.13** and uses `uv` for environment management.

### Install uv

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

Navigate to the project directory and run:

```bash
uv sync
```

This automatically creates a virtual environment (`.venv`) and installs the required dependencies, including PyTorch and NumPy.

## 2. Run Experiments

### Standard DE

Run DE on the default BBOB-40D function:

```bash
uv run main.py --algorithm de
```

Specify the function, evaluation budget, and random seed:

```bash
uv run main.py --algorithm de --function 8 --max-fes 10000 --seed 2024
```

### LLMDE

**Offline Test (Mock Controller)**

Test the LLMDE pipeline without API calls:

```bash
uv run main.py --algorithm llmde
```

**Live LLM Controller**

First, create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Configure the following environment variables in `.env`:

```env
GLM_CHAT_COMPLETIONS_URL=https://your-api-endpoint/chat/completions
GLM_MODEL=your-model-name
GLM_API_KEY=your-api-key
```

Then run LLMDE:

```bash
uv run --env-file .env main.py --algorithm llmde --function 8 --live --request-budget 45
```

## 3. Default Settings

| Parameter | Default |
|---|---|
| Benchmark | BBOB-40D (F1–F24) |
| Dimension | 40 |
| Population size | 50 |
| Maximum evaluations | 10,000 |
| Number of groups | 3 |
| Controller interval | 5 generations |
| Random seed | 2024 |
| DE parameters | F = 0.5, CR = 0.9 |

The BBOB benchmark uses fixed problem instances stored in `bbob40d.pt`. This file must remain in the project directory.

For additional command-line options, run:

```bash
uv run main.py --help
```
