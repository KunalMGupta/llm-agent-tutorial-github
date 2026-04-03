---
layout: default
title: "Nerfify: Multi-Agent in Practice"
nav_order: 11
---

# Nerfify: Multi-Agent System in Practice
<br>

Now let's see everything we've learned come together in a real project. **Nerfify** is a CrewAI-powered multi-agent pipeline that reads a research paper (PDF) and generates a complete, runnable [NeRFStudio](https://nerf.studio/) implementation — from parsing the paper all the way to smoke-testing the generated code.

<br>
<p style="text-align: center;">
  <img src="assets/nerfify-pipeline.png" alt="nerfify-pipeline" style="width: 90%;">
</p>
<br>

---

## The Problem
<br>

Implementing a NeRF paper from scratch is tedious:
1. Read the paper, extract the math and architecture details
2. Figure out which parts are novel vs. borrowed from cited works
3. Map the method onto the NeRFStudio template structure (config, model, field, pipeline, etc.)
4. Write all the code, making sure imports, loss functions, and forward passes are correct
5. Debug until it actually trains

Nerfify automates this entire pipeline using **6 specialized agents** orchestrated by CrewAI.

---

## The Agents
<br>

Each agent has a clear **role**, a set of **tools**, and a specific **task** in the pipeline:

| Agent | Role | Tools | Job |
|---|---|---|---|
| **Parser** | Paper Parser | `shell_command`, `clean_paper`, `web_search`, `file_read/write` | Download PDF, extract markdown via mineru, clean for implementation |
| **Citation Recovery** | Citation Specialist | `web_search`, `web_fetch`, `file_read/write` | Find implementation details missing from the paper but described in cited works |
| **Planner** | Architecture Planner | `web_search`, `file_read/write` | Design a dependency DAG and file generation plan |
| **Coder** | Senior NeRFStudio Engineer | `file_read/write`, `file_glob`, `shell_command`, `web_search` | Generate all 8 implementation files |
| **Reviewer** | Code Reviewer | `file_read/write`, `file_glob`, `web_search` | Review generated code for correctness |
| **Tester** | Smoke Tester | `shell_command`, `file_read` | Install, import-check, and run 10-iteration training |
| **Debugger** | Debug Specialist | `file_read/write`, `shell_command`, `web_search` | Diagnose and fix errors from failed tests |

---

## Pipeline Flow
<br>

The pipeline runs as a **sequential CrewAI process** — each task completes before the next begins (with one parallel exception):

```
┌──────────┐     ┌────────────────────┐     ┌──────────┐
│  Parser  │────▶│  Citation Recovery │────▶│ Planner  │
│          │     │  (runs in parallel │     │          │
│          │     │   with Planner)    │     │          │
└──────────┘     └────────────────────┘     └──────────┘
                                                  │
                        ┌─────────────────────────┘
                        ▼
                  ┌──────────┐     ┌──────────┐     ┌──────────┐
                  │  Coder   │────▶│ Reviewer │────▶│  Coder   │
                  │ (generate│     │ (review) │     │  (fix)   │
                  │  code)   │     │          │     │          │
                  └──────────┘     └──────────┘     └──────────┘
                                                         │
                        ┌────────────────────────────────┘
                        ▼
                  ┌──────────┐     ┌──────────┐
                  │  Tester  │────▶│ Debugger │ ←── loops up to 3x
                  │ (smoke   │     │ (fix     │
                  │  test)   │     │  errors) │
                  └──────────┘     └──────────┘
```

---

## How It's Built with CrewAI
<br>

Let's walk through the actual code.

### 1. Defining Tools

Each tool extends CrewAI's `BaseTool`. Here's the `ShellTool` that lets agents run system commands:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import subprocess

class ShellToolSchema(BaseModel):
    command: str = Field(default="", description="The shell command to execute")

class ShellTool(BaseTool):
    name: str = "shell_command"
    description: str = (
        "Execute a shell command and return the output. Use this for running "
        "mineru, pip install, ns-train, and other system commands."
    )
    args_schema: type = ShellToolSchema
    timeout: int = 600  # 10 minute timeout

    def _run(self, command: str = "") -> str:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=self.timeout,
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"EXIT CODE: {result.returncode}"
        return output or "(no output)"
```

And the `WebSearchTool` that uses DuckDuckGo:

```python
from ddgs import DDGS

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for information. Returns titles, URLs, and snippets."
    )

    def _run(self, query: str) -> str:
        results = DDGS().text(query, max_results=8)
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. **{r['title']}**\n   URL: {r['href']}\n   {r['body']}")
        return "\n\n".join(output)
```

And a domain-specific tool — `CleanPaperTool` — that uses an LLM to intelligently clean raw paper markdown:

```python
class CleanPaperTool(BaseTool):
    name: str = "clean_paper"
    description: str = (
        "Clean raw research paper markdown using an LLM. "
        "Removes narrative, keeps implementation details, preserves math."
    )
    model: str = "anthropic/claude-sonnet-4-20250514"

    def _run(self, input_path: str, output_path: str) -> str:
        raw_text = Path(input_path).read_text()

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": CLEANER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Clean this markdown:\n\n{raw_text}"},
            ],
        )

        cleaned = response.choices[0].message.content
        Path(output_path).write_text(cleaned)
        return f"Cleaned: {len(raw_text)} chars → {len(cleaned)} chars"
```

Notice that tools can themselves call LLMs — the `clean_paper` tool offloads large-context work to a dedicated LLM call rather than burdening the agent's conversation.

### 2. Defining Agents

Each agent gets a **role**, **goal**, **backstory** (system prompt), **tools**, and an **LLM model**:

```python
from crewai import Agent

# Shared tool instances
web_search = WebSearchTool()
file_read = FileReadTool()
file_write = FileWriteTool()
shell = ShellTool()
clean_paper = CleanPaperTool()

# Different agents get different tool sets
parser = Agent(
    role="Paper Parser",
    goal="Extract a research paper PDF to clean markdown suitable for implementation.",
    backstory="""You are the Paper Parser agent. Given a paper (PDF or arXiv URL):
    1. Download the PDF if needed
    2. Run mineru to extract markdown
    3. Clean the markdown using the clean_paper tool
    4. Scan references for implementation-critical resources""",
    tools=[file_read, file_write, shell, web_search, clean_paper],
    llm="anthropic/claude-haiku-4-5-20251001",  # Cheap model for parsing
    verbose=True,
    max_iter=10,
)

coder = Agent(
    role="Senior NeRFStudio Engineer",
    goal="Generate a complete, working NeRFStudio implementation from the paper.",
    backstory="""You are a senior NeRFStudio engineer. Generate all 8 required files.
    You MUST use file_write to save each file to disk.
    Match Nerfstudio APIs exactly. No placeholders, no TODOs.""",
    tools=[file_read, file_write, FileGlobTool(), shell, web_search],
    llm="anthropic/claude-sonnet-4-20250514",  # Best model for code generation
    verbose=True,
    max_iter=15,  # Needs more iterations to read templates + write 8 files
)
```

**Key design choice:** Different agents use different LLM models based on task complexity. The parser uses a cheaper/faster model (Haiku), while the coder uses the most capable one (Sonnet).

### 3. Defining Tasks with Dependencies

Tasks are where the actual work is described. The `context` parameter creates dependencies:

```python
from crewai import Task

parse_task = Task(
    description=f"""Parse the research paper and produce clean markdown.
    Input: {paper_input}
    Workspace: {workspace}
    Steps:
    1. Download the PDF: shell_command("wget -O {workspace}/paper.pdf <url>")
    2. Run mineru: shell_command("mineru -p <pdf> -o {workspace}/mineru_output")
    3. Clean: clean_paper(input="{workspace}/raw_paper.md", output="{workspace}/cleaned_paper.md")
    4. Scan references with web_search""",
    expected_output="Cleaned paper markdown saved to workspace.",
    agent=parser,
)

# Citation recovery runs in PARALLEL with planning
citation_task = Task(
    description="Recover implementation details from cited papers...",
    agent=citation_recovery_agent,
    async_execution=True,  # ← Runs in parallel!
)

plan_task = Task(
    description="Create an architecture plan for the NeRFStudio implementation...",
    agent=planner,
    async_execution=True,  # ← Also runs in parallel with citation
)

# Code generation WAITS for both parallel tasks to complete
code_task = Task(
    description=f"""Generate the complete NeRFStudio implementation.
    Read: {workspace}/cleaned_paper.md, {workspace}/dag_plan.json
    Output: {output_dir}
    Use file_write to create ALL 8 files...""",
    expected_output="All 8 files written to disk. Self-check passed.",
    agent=coder,
    # Implicitly waits for citation_task and plan_task since they're earlier in the list
)
```

### 4. Assembling the Crew

Finally, everything comes together in the `Crew`:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[parser, citation_recovery, planner, coder, reviewer, tester, debugger],
    tasks=[parse_task, citation_task, plan_task, code_task, review_task, 
           fix_task, test_task, debug_task_1, debug_task_2, debug_task_3],
    process=Process.sequential,  # Tasks run in order (respecting async_execution)
    verbose=True,
    memory=False,
    full_output=True,
)

# Kick it off!
result = crew.kickoff()
```

### 5. Running the Pipeline

```bash
# From an arXiv paper
python main.py --arxiv 2308.12345

# With options
python main.py --arxiv 2308.12345 --method-name my_nerf --train --gpu 0

# Fast mode (skip citation recovery)
python main.py --pdf paper.pdf --fast --no-review
```

---

## Tool Usage in Detail
<br>

Let's trace how tools get used across the pipeline:

### Parser Agent Tool Calls

```
1. shell_command("wget -O workspace/paper.pdf https://arxiv.org/pdf/2308.12345.pdf")
   → Downloads the PDF

2. shell_command("mineru -p workspace/paper.pdf -o workspace/mineru_output")
   → Extracts raw markdown from PDF

3. clean_paper(input_path="workspace/raw_paper.md", output_path="workspace/cleaned_paper.md")
   → LLM-powered cleaning: 50,000 chars → 20,000 chars of pure implementation detail

4. web_search("NeRF hashgrid implementation GitHub")
   → Finds referenced repos and implementations
```

### Coder Agent Tool Calls

```
1. file_read("base-code/method_template/template_model.py")
   → Reads the NeRFStudio template to match the API

2. file_read("workspace/dag_plan.json")
   → Reads the architecture plan

3. web_search("nerfstudio Field class forward method signature")
   → Verifies API details

4. file_write("output/method_template/template_model.py", content="...")
   → Writes the generated model file

5. file_write("output/method_template/template_field.py", content="...")
   → Writes the field file
   ... (repeats for all 8 files)

6. file_glob(pattern="**/*", directory="output/")
   → Verifies all files were written to disk
```

### Tester Agent Tool Calls

```
1. shell_command("conda activate nerfstudio; cd output && pip install -e .")
   → Installs the generated package

2. shell_command("conda activate nerfstudio; cd output && ns-install-cli")
   → Registers the CLI commands

3. shell_command("conda activate nerfstudio; python -c 'from method_template.template_config import method_template'")
   → Verifies all imports resolve

4. shell_command("conda activate nerfstudio; ns-train my_nerf --data data/chair --max-num-iterations 10")
   → Runs a 10-iteration smoke test to verify the training loop works
```

---

## What Makes This a Good Multi-Agent Design?
<br>

1. **Separation of concerns** — Each agent is an expert at one thing. The parser doesn't write code; the coder doesn't debug.

2. **Right model for the job** — Cheap models (Haiku) for mechanical tasks like parsing. Expensive models (Sonnet) for code generation. This saves cost without sacrificing quality where it matters.

3. **Tool specialization** — The tester only gets `shell_command` and `file_read`. It can't accidentally modify code. The coder gets `file_write` but no shell access (until the fix phase).

4. **Feedback loops** — The review → fix cycle catches errors before testing. The test → debug cycle catches runtime issues. Up to 3 debug iterations prevent infinite loops.

5. **Parallel execution** — Citation recovery and planning run simultaneously since they're independent, cutting wall-clock time.

---

📚 **References**

1. [CrewAI Documentation](https://docs.crewai.com/)
2. [NeRFStudio](https://nerf.studio/)
3. [Nerfify Project Page](https://seemandhar.github.io/NERFIFY/)

---

## About the Author

**Kunal Gupta**  
[Website](https://kunalmgupta.github.io)  
[Email](mailto:k5gupta@ucsd.edu)  
[GitHub](https://github.com/KunalMGupta)
