import os
import sys
from pathlib import Path
import datetime

from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent
from google.adk.tools import agent_tool
from google.adk.tools.tool_context import ToolContext
from google.adk.models.lite_llm import LiteLlm

# ── env/config ───────────────────────────────────────────────────────────────
load_dotenv()

MODEL = LiteLlm(
   model=os.getenv(
      "MODEL",
      "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
   )
)


# ── Shared tool: deterministic loop exit ────────────────────────────────────
def exit_loop(tool_context: ToolContext) -> dict:
   """Call this tool ONLY when validation passes, to end the current
   refinement loop immediately instead of burning remaining iterations."""
   tool_context.actions.escalate = True
   # NOTE: do NOT set skip_summarization here. AgentTool (which wraps this
   # LoopAgent for the root agent) returns whatever text was in the LAST
   # event of the run, and a function_response event carries no extractable
   # text. Suppressing the follow-up model turn left that response empty,
   # so the root agent saw the writer/planner tool "return nothing." This
   # costs one extra LLM call per successful stage vs. the ideal, but it's
   # required for the tool response to carry any text at all.
   return {}


# ── Sub-Agent: Planner ───────────────────────────────────────────────────────
blog_planner = Agent(
   name="BlogPlanner",
   model=MODEL,
   description="Creates a practical, skimmable outline in Markdown.",
   instruction="""
You are a technical content strategist. Produce a clear Markdown outline with:
- Title
- Short intro
- 4–6 main sections (each with 2–3 bullets)
- Conclusion

If `codebase_context` exists in state, weave in specific sections/snippets.
Return only the outline in Markdown.
""",
   output_key="blog_outline",
)

class OutlineValidationChecker(Agent):
   def __init__(self):
       super().__init__(
           name="OutlineValidationChecker",
           model=MODEL,
           description="Validates that the outline is usable.",
           instruction="""
Check the outline in state `blog_outline`.

If it has a title, intro, 4–6 sections, and a conclusion:
- Call the `exit_loop` tool immediately, with no other text.

Otherwise:
- Do NOT call `exit_loop`.
- Respond with the exact word "retry" followed by a list of the
  specific pieces that are missing, so the planner can fix them
  on the next attempt.
""",
           tools=[exit_loop],
           output_key="outline_validation_result",
       )

robust_blog_planner = LoopAgent(
   name="RobustBlogPlanner",
   description="Retries planning if validation fails.",
   sub_agents=[blog_planner, OutlineValidationChecker()],
   max_iterations=3,
)

# ── Sub-Agent: Writer ────────────────────────────────────────────────────────
blog_writer = Agent(
   name="BlogWriter",
   model=MODEL,
   description="Writes a technical blog post from the outline.",
   instruction="""
Write a complete Markdown article from the outline in `blog_outline`.

Guidelines:
- Audience: software engineers; skip basics and focus on practical insight.
- Explain both the 'how' and 'why'.
- Include concise code snippets when helpful.
- Follow the outline’s structure (H2/H3).
- Output only the final article in Markdown (no fence around the whole post).
""",
   output_key="blog_post",
)

class BlogPostValidationChecker(Agent):
   def __init__(self):
       super().__init__(
           name="BlogPostValidationChecker",
           model=MODEL,
           description="Validates the final post.",
           instruction="""
Check `blog_post` for: intro, clear sections matching the outline, conclusion,
and technical clarity.

If it passes:
- Call the `exit_loop` tool immediately, with no other text.

Otherwise:
- Do NOT call `exit_loop`.
- Respond with the exact word "retry" followed by the specific fixes needed.
""",
           tools=[exit_loop],
           output_key="post_validation_result",
       )

robust_blog_writer = LoopAgent(
   name="RobustBlogWriter",
   description="Retries writing if validation fails.",
   sub_agents=[blog_writer, BlogPostValidationChecker()],
   max_iterations=3,
)

# Expose planner/writer as tools so the root agent can call them explicitly
planner_tool = agent_tool.AgentTool(agent=robust_blog_planner)
writer_tool  = agent_tool.AgentTool(agent=robust_blog_writer)

# ── Root Agent: Plan → Write ────────────────────────────────────────────────
root_agent = Agent(
   name="Blogger",
   model=MODEL,
   description="Minimal multi-agent blogger that plans and writes.",
   instruction=f"""
If the user gives a topic:
1) Call the planner tool to generate the outline.
2) Call the writer tool to produce the full draft.
3) End with 3 alternate titles and 2 tweet-length hooks.

Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
""",
   tools=[
       planner_tool, # calls RobustBlogPlanner
       writer_tool,  # calls RobustBlogWriter
   ],
)
