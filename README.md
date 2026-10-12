# neo-analyst

**Not your regular Applied ML project *(or maybe)*.**

A tool-using data analysis agent that answers natural-language questions about CSV datasets using Python-based analytics and an LLM.

## Features
- Six analytical tools: overview, statistics, filtering, correlation, visualization, and aggregation.
- Multi-step LLM agent that selects and executes tools.
- Grounding validator for checking numerical claims and contradictions.
- Streamlit interface for uploading datasets and exploring results.

## Tech Stack
Python, Pandas, NumPy, Plotly, OpenAI API, Streamlit, Pytest.

## Project structure

```
neo-analyst/
├── app/
│   ├── analysis/       # Six analytical tools
│   ├── agent/          # LLM agent and grounding validator
│   └── ui/             # Streamlit interface
├── examples/           # Sample datasets and demos
├── tests/              # Automated tests
├── requirements.txt
├── pyproject.toml
└── README.md
```
---

## Setup

```bash
git clone https://github.com/neolakshay/neo-analyst.git
cd neo-analyst

python -m venv .venv
```

Activate the environment:

**Linux/macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL` is optional and defaults to `gpt-4o-mini`. The agent and web interface require a valid OpenAI API key.

## Running the Project

```bash
# Analytical engine demo
python -m examples.demo

# LLM agent demo
python -m examples.agent_demo "Which department has the highest average salary?"

# Web interface
python -m streamlit run app/ui/app.py
```
