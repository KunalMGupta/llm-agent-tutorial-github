---
layout: default
title: Open-Source Frameworks
nav_order: 11
---

# Open-Source Multi-Agent Frameworks
<br>

Building multi-agent systems from scratch (like we did with Agent252D) teaches you the fundamentals, but in practice you'll want a framework that handles **orchestration**, **tool routing**, **memory**, and **agent communication** for you. Let's look at the three most popular options.

<div style="display:flex;gap:1.2rem;justify-content:center;flex-wrap:wrap;margin:2rem 0 1.5rem">
  <div style="text-align:center;padding:1.2rem 1.5rem;border-radius:12px;background:linear-gradient(135deg,#1a1a2e,#16213e);min-width:180px;flex:1;max-width:240px;border:1px solid #2a2a4a">
    <div style="font-size:2.5rem;margin-bottom:0.4rem">🚢</div>
    <div style="font-weight:700;font-size:1.1rem;color:#e94560">CrewAI</div>
    <div style="font-size:0.75rem;color:#a0a0b8;margin-top:0.3rem">Role-based agent crews</div>
  </div>
  <div style="text-align:center;padding:1.2rem 1.5rem;border-radius:12px;background:linear-gradient(135deg,#1a1a2e,#16213e);min-width:180px;flex:1;max-width:240px;border:1px solid #2a2a4a">
    <div style="font-size:2.5rem;margin-bottom:0.4rem">🔗</div>
    <div style="font-weight:700;font-size:1.1rem;color:#00b4d8">LangGraph</div>
    <div style="font-size:0.75rem;color:#a0a0b8;margin-top:0.3rem">State machine workflows</div>
  </div>
  <div style="text-align:center;padding:1.2rem 1.5rem;border-radius:12px;background:linear-gradient(135deg,#1a1a2e,#16213e);min-width:180px;flex:1;max-width:240px;border:1px solid #2a2a4a">
    <div style="font-size:2.5rem;margin-bottom:0.4rem">🤖</div>
    <div style="font-weight:700;font-size:1.1rem;color:#d4a574">Claude Agent SDK</div>
    <div style="font-size:0.75rem;color:#a0a0b8;margin-top:0.3rem">Native tool-use agents</div>
  </div>
</div>

---

## What Do These Frameworks Give You?
<br>

| Concern | Rolling Your Own | Using a Framework |
|---|---|---|
| Agent ↔ Agent communication | You write the message passing | Built-in (delegation, context passing) |
| Tool integration | Manual prompt injection | Declarative tool registration |
| Execution order | Hardcoded or ad-hoc | Sequential, hierarchical, or graph-based |
| Error handling & retries | Manual try/except | Configurable retry policies |
| Observability | Print statements | Built-in logging, token tracking |

---

## 1. CrewAI
<br>

**Philosophy:** *"A crew of AI agents, each with a role, working together on tasks."*

CrewAI is the most beginner-friendly framework. You define **Agents** (who they are) and **Tasks** (what they do), then assemble them into a **Crew** that runs sequentially or hierarchically.

### Core Concepts

```
Agent  ─── has ──→  Role, Goal, Backstory, Tools, LLM
Task   ─── has ──→  Description, Expected Output, Agent, Context
Crew   ─── has ──→  [Agents], [Tasks], Process (sequential | hierarchical)
```

### Minimal Example

```python
from crewai import Agent, Task, Crew, Process

# 1. Define agents with roles and tools
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive information about a given topic",
    backstory="You are an expert researcher with 20 years of experience.",
    tools=[web_search_tool, web_fetch_tool],  # Tools the agent can use
    llm="anthropic/claude-sonnet-4-20250514",
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, engaging summaries from research findings",
    backstory="You are a skilled writer who makes complex topics accessible.",
    tools=[file_write_tool],
    llm="anthropic/claude-sonnet-4-20250514",
    verbose=True,
)

# 2. Define tasks (what needs to be done)
research_task = Task(
    description="Research the latest advances in Neural Radiance Fields (NeRF).",
    expected_output="A detailed report with key papers, methods, and trends.",
    agent=researcher,
)

writing_task = Task(
    description="Write a blog post summarizing the research findings.",
    expected_output="A 500-word blog post in markdown format.",
    agent=writer,
    context=[research_task],  # This task depends on research_task
)

# 3. Assemble the crew and run
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # Tasks run in order
    verbose=True,
)

result = crew.kickoff()
print(result)
```

### How Tools Work in CrewAI

Tools in CrewAI extend the `BaseTool` class. The framework automatically injects tool descriptions into the agent's prompt, and the agent decides when to call them:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchSchema(BaseModel):
    query: str = Field(..., description="The search query")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information. Returns titles, URLs, and snippets."
    args_schema: type = SearchSchema

    def _run(self, query: str) -> str:
        # Your search implementation here
        results = search_engine.search(query, max_results=5)
        return format_results(results)
```

The agent sees the tool name + description in its system prompt and generates structured tool calls when needed. CrewAI handles parsing the LLM output and routing to the right tool.

### When to Use CrewAI

- Straightforward pipelines where agents run **sequentially** or in a **supervisor hierarchy**
- Rapid prototyping — least boilerplate of the three frameworks
- When you want role-based agent design (each agent has a clear persona)

---

## 2. LangGraph
<br>

**Philosophy:** *"Agents as nodes in a state machine graph."*

LangGraph (from LangChain) models your multi-agent system as a **directed graph**. Each node is an agent or function, edges define transitions, and a shared **state** object flows through the graph. This gives you fine-grained control over execution flow, including loops, conditionals, and parallel branches.

### Core Concepts

```
State       ─── shared dict passed between nodes
Node        ─── a function or agent that reads/writes state
Edge        ─── transition (can be conditional)
Graph       ─── the full workflow, compiled into a runnable
```

### Minimal Example

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# 1. Define shared state
class ResearchState(TypedDict):
    topic: str
    research: Annotated[list[str], operator.add]  # Accumulates results
    draft: str
    feedback: str

# 2. Define node functions (each is an "agent")
def researcher(state: ResearchState) -> dict:
    """Research node — calls an LLM with search tools."""
    result = llm.invoke(f"Research: {state['topic']}")
    return {"research": [result.content]}

def writer(state: ResearchState) -> dict:
    """Writer node — drafts content from research."""
    research = "\n".join(state["research"])
    draft = llm.invoke(f"Write a summary based on:\n{research}")
    return {"draft": draft.content}

def reviewer(state: ResearchState) -> dict:
    """Reviewer node — checks the draft quality."""
    feedback = llm.invoke(f"Review this draft:\n{state['draft']}")
    return {"feedback": feedback.content}

def should_revise(state: ResearchState) -> str:
    """Conditional edge: route based on review feedback."""
    if "approved" in state["feedback"].lower():
        return "end"
    return "revise"

# 3. Build the graph
graph = StateGraph(ResearchState)

graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")

# Conditional: reviewer can approve or send back for revision
graph.add_conditional_edges("reviewer", should_revise, {
    "revise": "writer",   # Loop back to writer
    "end": END,
})

# 4. Compile and run
app = graph.compile()
result = app.invoke({"topic": "Neural Radiance Fields", "research": []})
print(result["draft"])
```

### When to Use LangGraph

- Complex workflows with **loops**, **conditionals**, or **parallel branches**
- When you need explicit control over **state** and **transitions**
- Systems that require **human-in-the-loop** checkpoints
- When you want to visualize the agent workflow as a graph

---

## 3. Claude Agent SDK (Anthropic)
<br>

**Philosophy:** *"Agents as Python classes with tool-use built into the model."*

The Claude Agent SDK is Anthropic's lightweight framework. Instead of complex orchestration, it leans on Claude's native tool-use capability. You define agents with tools, and the model decides when and how to use them in a natural conversation loop.

### Core Concepts

```
Agent       ─── name, model, instructions, tools, handoffs
Tool        ─── a Python function decorated with @tool
Handoff     ─── transfers control from one agent to another
Runner      ─── executes the agent loop (handles tool calls automatically)
```

### Minimal Example

```python
from claude_agent_sdk import Agent, tool, Runner

# 1. Define tools as decorated functions
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return search_engine.search(query)

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file on disk."""
    Path(path).write_text(content)
    return f"Written to {path}"

# 2. Define agents
researcher = Agent(
    name="Researcher",
    model="claude-sonnet-4-20250514",
    instructions="You are a research specialist. Find information using web search.",
    tools=[web_search],
)

writer = Agent(
    name="Writer",
    model="claude-sonnet-4-20250514",
    instructions="You are a technical writer. Write clear summaries and save to files.",
    tools=[write_file],
    handoffs=[researcher],  # Can hand off to researcher if more info needed
)

# 3. Run the agent
result = Runner.run(writer, "Write a blog post about NeRF advances")
print(result.final_output)
```

### When to Use Claude Agent SDK

- When you're already using Claude and want tight integration
- Simpler agent setups where **handoffs** between agents suffice
- When you want the model's native tool-use rather than prompt-engineered parsing
- Lightweight projects — minimal dependencies, no heavy framework overhead

---

## Framework Comparison
<br>

| Feature | CrewAI | LangGraph | Claude Agent SDK |
|---|---|---|---|
| **Learning Curve** | Low | Medium-High | Low |
| **Execution Model** | Sequential / Hierarchical | State machine graph | Conversation loop with handoffs |
| **Tool Integration** | BaseTool classes | LangChain tools or functions | `@tool` decorated functions |
| **State Management** | Implicit (task context) | Explicit (TypedDict state) | Conversation history |
| **Loops & Conditionals** | Limited (via task design) | First-class (conditional edges) | Via agent instructions |
| **Parallel Execution** | `async_execution=True` on tasks | Parallel branches in graph | Multiple agents via handoffs |
| **Best For** | Role-based pipelines | Complex stateful workflows | Claude-native lightweight agents |

---

## How Tools Work Across Frameworks
<br>

Tools are the **hands** of an agent — they let the LLM interact with the real world. Every framework supports similar tool types:

### Common Tool Categories

| Tool Type | What It Does | Example |
|---|---|---|
| **Web Search** | Query search engines | `web_search("NeRF tutorial")` |
| **Web Fetch** | Download and read web pages | `web_fetch("https://arxiv.org/abs/...")` |
| **File Read/Write** | Read and write local files | `file_read("paper.md")`, `file_write("output.py", code)` |
| **Shell Command** | Execute system commands | `shell("pip install nerfstudio")` |
| **Code Execution** | Run Python code in a sandbox | `run_python("print(2+2)")` |
| **API Calls** | Call external APIs (Slack, GitHub, etc.) | `github_create_issue(title, body)` |
| **Custom Domain Tools** | Domain-specific operations | `clean_paper(input, output)`, `train_model(config)` |

### How the LLM Uses Tools

The flow is the same regardless of framework:

```
1. Agent receives a task/prompt
2. LLM generates a response — may include a tool call
   e.g., "I need to search for this paper" → calls web_search(...)
3. Framework intercepts the tool call, executes the function
4. Tool result is fed back to the LLM as context
5. LLM continues reasoning with the new information
6. Repeat until task is complete (or max iterations reached)
```

The framework's job is to handle steps 2-5 automatically — parsing the LLM's tool calls, routing to the right function, and feeding results back.

---

📚 **References**

1. [CrewAI Documentation](https://docs.crewai.com/)
2. [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
3. [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk)

---

## Project Ideas: Multi-Agent Systems for Computer Vision
<br>

These frameworks really shine when applied to **computer vision** pipelines — tasks that are naturally decomposable into specialized stages. Here are project ideas at different difficulty levels, each designed around the multi-agent patterns we've covered.

### 1. Paper2Code — Research Paper to Working Implementation

> **Example:** [Nerfify](nerfify) — converts a vision paper (PDF) into a complete, trainable NeRFStudio codebase.

| Agent | Role | Framework Pattern |
|---|---|---|
| **Parser** | Extract equations, architecture, and pseudocode from a PDF | Tool-use (shell, LLM cleaning) |
| **Planner** | Design code architecture and dependency graph | Sequential handoff |
| **Coder** | Generate implementation files | Heavy tool-use (file write, web search) |
| **Tester** | Run smoke tests, catch shape mismatches | Shell execution |
| **Debugger** | Diagnose and fix errors from test output | Feedback loop (test ↔ debug) |
| **VLM Evaluator** | Render outputs, use a vision-language model to judge quality | Conditional loop (retrain if quality fails) |

**Why it works as a multi-agent project:** Each stage requires different expertise and tools. The feedback loops (test→debug, VLM→retrain) map directly to LangGraph conditional edges or CrewAI task dependencies.

---

### 2. Scene Understanding & 3D Reconstruction Pipeline

Build an end-to-end system that takes casual photos and produces a 3D reconstruction with quality guarantees.

| Agent | Role |
|---|---|
| **Capture Validator** | Check input views for sufficient overlap, blur, and exposure |
| **Depth Estimator** | Run monocular depth (ZoeDepth / Depth Anything) |
| **Segmenter** | Run SAM2 to extract object masks |
| **Reconstructor** | Run SfM + 3DGS/NeRF on the scene |
| **Quality Inspector (VLM)** | Compare rendered novel views against inputs, flag artifacts |
| **Refinement Agent** | Adjust hyperparameters or mask problem regions and re-train |

**Framework fit:** LangGraph — the Quality Inspector creates a conditional loop back to Refinement, and the first three agents can run as parallel branches.

---

### 3. Automated Dataset Curation & Annotation

Every CV project starts with data. Build agents that assemble a publication-ready dataset from scratch.

| Agent | Role |
|---|---|
| **Scraper** | Collect images from web/videos given a text description |
| **Filter** | Remove duplicates, blurry, or irrelevant images (CLIP-based) |
| **Annotator** | Run detection/segmentation models for initial labels |
| **VLM Verifier** | Review annotations using a vision-language model, flag errors |
| **Human-in-the-Loop** | Present uncertain cases for human review |
| **Exporter** | Package into COCO/YOLO format with train/val/test splits and stats |

**Framework fit:** CrewAI — straightforward sequential pipeline with clear role-based agents. Good starter project.

---

### 4. Visual Debugging Agent for Training Runs

Automate what CV researchers do manually: watch training, spot problems, and fix them.

| Agent | Role |
|---|---|
| **Monitor** | Watch TensorBoard/W&B logs for anomalies (loss spikes, NaN, plateau) |
| **Visualizer** | Render intermediate predictions at checkpoints |
| **Diagnosis (LLM)** | Analyze loss curves + visual outputs, hypothesize issues |
| **Fix Agent** | Modify config (learning rate, augmentation, etc.) and restart training |
| **Comparison** | Side-by-side eval of runs, pick best checkpoint |

**Framework fit:** LangGraph — the Monitor→Diagnosis→Fix→Monitor loop is a natural state machine with conditional edges for different failure modes.

---

### 5. Interior Design / Room Staging Agent

Combine 3D understanding with generative models for a creative, visually impressive application.

| Agent | Role |
|---|---|
| **Room Understanding** | Segment room layout, identify furniture, estimate geometry |
| **Style Agent (LLM)** | Propose design changes based on user preferences |
| **Generation** | Use inpainting/ControlNet to render proposed changes |
| **Consistency Checker (VLM)** | Verify lighting, perspective, and style consistency across edits |
| **Iteration Agent** | Take user feedback, refine the design |

**Framework fit:** Claude Agent SDK — the conversational handoff pattern works well for iterative user-facing design workflows.

---

### Choosing a Project by Experience Level

| Experience | Recommended Project | Why |
|---|---|---|
| **Beginner** | Dataset Curation (#3) | Concrete, useful, lower barrier to entry |
| **Intermediate** | Visual Debugging (#4) or Interior Design (#5) | Teaches real training workflows or creative generation |
| **Advanced** | Paper2Code (#1) or 3D Reconstruction (#2) | Full research pipeline, touches NLP + vision + evaluation |

---

## 🧭 What's Next?
Let's see a real multi-agent system in action! We'll look at **Nerfify** — a CrewAI pipeline that converts research papers into working code.

➡️ [Nerfify: Multi-Agent in Practice](nerfify)
