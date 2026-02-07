# Expert Parallelism Load Balancer (EPLB)

This example demonstrates how to use OpenEvolve to optimize the Expert Parallelism Load Balancer (EPLB) algorithm.

## Setup

Install PyTorch:

```bash
uv pip install torch
```

Download the workload file from [Hugging Face](https://huggingface.co/datasets/abmfy/eplb-openevolve):

```bash
wget https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json
```


# Role Definition
You are an expert AI Researcher and Software Engineer specializing in Evolutionary Algorithms (EA), Large Language Models (LLM), and Reinforcement Learning (RL).

# Objective
We are going to refactor the `ADRS` repository. The goal is to upgrade the standard evolutionary framework into a **Test-Time RL-enhanced Evolutionary System** (integrate some advanced features from RL which evolutionary algorithms are missing).

We will achieve this by injecting reasoning capabilities (Summarization, Criticism/Gradient, and History Replay) into the evolution loop.

# Prerequisite: Codebase Exploration
First, please scan the repository structure, specifically looking for:
1. The class definition for an Individual or Program.
2. The main evolution loop, specifically where `mutate(parent)` is called.
3. The LLM interface/client.

# Implementation Plan
Please modify the codebase to implement the following 6 specific features.

## Feature 1: Data Structure Expansion (The "State")
Modify the Program/Individual class. It must store following attributes:
- `abstract` (str): A high-level summary of the algorithm's logic.
- `gradient` (str): Textual feedback/criticism analyzing improvements needed for this specific program.
- `provenance` (dict/str): Metadata tracking lineage.

## Feature 2: The "Summary Agent" & "Critic Agent"
Create a new module (e.g., `rl_agents.py`) containing the prompt logic for two new LLM calls that occur **immediately after** a child program is generated and evaluated:
1.  **Summary Agent:**
    -   **Input:** The `Child Program Code`.
    -   **Output:** A concise `abstract` describing the core algorithmic idea.
    -   **Action:** Save this to `child.abstract`.
2.  **Critic Agent (The Textual Gradient):**
    -   **Input:** The `Child Program Code`, its `Abstract`, and evaluation logs.
    -   **Output:** A `gradient` (textual analysis) identifying specific weaknesses (e.g., "Runtime is O(N^2), needs O(N)").
    -   **Action:** Save this to `child.gradient`.

## Feature 3: The "History Buffer" (Replay Memory)
Implement a lightweight storage system to log the evolution trajectory.
-   **Data to Store:** `(Parent Abstract) -> (Child Abstract)` mapped to `(Fitness Delta)`.
-   **Goal:** To serve as few-shot examples of successful logic shifts.

## Feature 4: The "Reference Selection Agent" (Context Retrieval)
**This is a new step that happens INSIDE the mutation loop, AFTER a parent has been selected by the standard EA logic.**
Create a function `get_references(parent, population, k=3)`:
-   **Input:**
    -   The `parent.gradient` (What needs fixing).
    -   The `parent.abstract` (What the parent currently does).
    -   A list of `abstracts` from the rest of the population (candidates).
-   **LLM Task:** "Given the parent's current logic and its gradient (needs for improvement), select K programs from the candidates that serve as the best references."
-   **Selection Criteria (Explicitly Instruct LLM):**
    1.  **Relevance:** Does the candidate have logic that solves the `gradient` issue?
    2.  **Quality:** Prefer candidates with high fitness.
    3.  **Diversity:** Do not select K identical programs; look for different approaches.
-   **Output:** A list of `{top_programs}` (code or abstracts) to be used as context.

## Feature 5: The "RL-Enhanced Evolver" (The Policy)
Modify the core `evolve` / `mutate` function. When asking the LLM to generate a new child from the **standardly selected parent**, the Prompt must now include:
1.  **The Parent Code.**
2.  **The Parent's Textual Gradient:** "The Critic pointed out these issues in the Parent: {parent.gradient}"
3.  **The Reference Programs (Top-K):** "Here are {K} other programs from the population selected because they might contain logic useful for fixing the gradient. Use them as inspiration but do not copy blindly: \n {top_programs}"
4.  **In-Context History (M-Shot):** Randomly sample `M` entries from the "History Buffer" where `Fitness Delta > 0` to show successful evolution patterns.

## Feature 6: Execution & Storage
-   Update the loop to ensure Features 1-5 run sequentially.
-   Save the `History Buffer` to a persistent file (e.g., `evolution_history.jsonl`) so it grows across generations.

# Execution Instructions
1.  **Analyze**: Confirm where `mutate` is called and how parents are currently passed to it.
2.  **Design**: Draft the prompt for the *Reference Selection Agent* (Feature 4) to ensure it balances relevance and diversity.
3.  **Refactor**: Apply the changes. Implement a switch to enable/disable this "RL Mode" so the original baseline can still run if needed.

Let's begin by analyzing the existing mutation logic.