# 🌾 Field Vision

## 📌 Introduction

**Field Vision** is a research and development project that brings together **Reinforcement Learning (RL)**, **Computer Vision (CV)**, and **Edge Computing** to optimize the  UAVs  during flight computation task in agricultural and field-based environments.

The goal is to:

- Use **computer Federated vision algorithms** for:
  - Accurate **homography estimation**.
  - **Plant counting** in aerial imagery.
- Implement **RL-based computation offloading** to balance onboard and cloud/edge processing for real-time operation during flight.


---

## 🗂 Repository Structure (Suggested)
```
field-vision/
│
├── README.md
├── pyproject.toml # For dependency management (Poetry or uv)
│
├── notebooks/ #  Script and Jupyter notebooks  for research and prototyping
│ ├── rl/
│ ├── vision/
│ └── offloading/
│
├── src/ # Core source code
│ ├── init.py
│ ├── rl/
│ ├── vision/
│ └── offloading/
│
├── tests/ # Unit and integration tests
│ ├── rl/
│ ├── vision/
│ └── offloading/
│
├── config/ # YAML/JSON config files for modularity
│
├── data/ # Sample or synthetic data (NO raw data committed)
│
├── scripts/ # CLI scripts for running training and inference
│
└── production/ # Docker, CI/CD, and deployment setup
├── Dockerfile
├── docker-compose.yml
├── start.sh
└── configs
```



---

## 🔧 Setup & Installation

Install dependencies using [Poetry](https://python-poetry.org/) or [uv](https://github.com/astral-sh/uv):

```bash
# Using Poetry
poetry install

# OR using uv
uv pip install -r requirements.txt
```

If you are familiar with TDD and want to write some test use pytest.

```
pytest tests/
```



## 👥 Contributing Guidelines

We welcome contributions! Please follow these conventions to maintain code quality and project consistency:

🧼 Code Style & Quality
Use Python 3.10+.
Follow PEP8 for formatting. Use tools like black, flake8, and isort.
Write type-annotated code wherever possible.


🧱 Project Architecture
Modular design: separate logic for RL, CV, and offloading.
Use configuration files (.yaml or .json) for hyperparameters, environment settings, etc. Avoid hardcoding parameters in scripts.
Do not commit raw data or .env files. Use data/README.md to describe data access.
Use logging instead of print statements in production code.


🧩 Dependency Management
Use Poetry or uv for consistent environment management.
Keep pyproject.toml up-to-date.
Never commit virtual environments or OS-specific files.


🌱 Git Workflow
Use feature branches (feature/, bugfix/, refactor/) for all work.
Keep develop and main branches clean and stable.
Open a Pull Request (PR) for all changes, no direct commits to main or develop.
Write meaningful commit messages and document your changes in PRs.
Invite at least one teammate for code review every week.
Rebase frequently and resolve conflicts proactively.
