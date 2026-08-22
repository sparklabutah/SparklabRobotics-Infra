"""High-level agents that decide WHAT the low-level VLA should be told to do.

The split this package exists to serve:

    goal ("clear the table")          <- you, or a caller
      -> high-level agent             <- reasons over camera frames, this file
      -> subtask ("pick up the scissors")
      -> MolmoAct2                    <- turns a task string + images into joints
      -> arms

The VLA already takes a natural-language task and re-plans from scratch when it
changes (``rollout_live.retarget()`` clears the action queue on every change).
So the entire integration surface for a high-level agent is: look at the scene,
emit a new task string when the current one is finished or wrong.

AN AGENT IMPLEMENTS ``decide(goal, frames, state, history) -> Decision``, and
optionally ``reset()``, which the server calls when the goal changes. Stateless
agents can omit it.

WHAT AN AGENT MAY DO. Emit a task string and declare the goal complete. That is
all. It cannot park, reset, or stop the robot -- those stay with the human at
the console. An agent that can retarget can already move the arms, so this is
not a security boundary; it is a blast-radius one. A confused agent picks the
wrong subtask, it does not decide to power down mid-motion.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """One agent turn."""

    task: str | None = None          # new subtask, or None to keep the current one
    done: bool = False               # the GOAL (not the subtask) is complete
    reasoning: str = ""              # shown in the UI trace
    error: str | None = None
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"task": self.task, "done": self.done, "reasoning": self.reasoning,
                "error": self.error, "at": self.at}


class ScriptedAgent:
    """Walks a fixed list of subtasks. No API key, no network.

    Here so the whole harness -- server, frontend, retargeting, the robot --
    can be exercised end to end before any model is wired in. If the arm does
    not do the right thing under this agent, the problem is not the model.
    """

    name = "scripted"

    def __init__(self, subtasks: list[str] | None = None, dwell_s: float = 12.0):
        self.subtasks = subtasks or [
            "pick up the marker",
            "put the marker into the cardboard box",
        ]
        self.dwell_s = dwell_s
        self._i = 0
        self._since = 0.0

    def reset(self) -> None:
        """Start the list again. Called when the goal changes.

        Without this, setting a new goal left the walker at the end of the old
        one: the next turn reported "list exhausted" with done=True, so Start
        armed the loop and the first turn immediately disarmed it.
        """
        self._i = 0
        self._since = 0.0

    def decide(self, goal, frames, state, history) -> Decision:
        now = time.time()
        if self._since and now - self._since < self.dwell_s:
            # self._i already points at the NEXT subtask, so the one currently
            # executing is _i - 1.
            current = self.subtasks[max(0, self._i - 1)]
            return Decision(reasoning=f"holding {current!r} "
                                      f"({now - self._since:.0f}/{self.dwell_s:.0f}s)")
        if self._i >= len(self.subtasks):
            return Decision(done=True, reasoning="scripted list exhausted")
        task = self.subtasks[self._i]
        self._i += 1
        self._since = now
        return Decision(task=task, reasoning=f"scripted step {self._i}/{len(self.subtasks)}")


_SYSTEM = """You direct a bimanual robot by issuing ONE short instruction at a time.

A low-level policy converts your instruction into joint motion. It understands
short concrete manipulation phrases like "pick up the red marker" or "put the
marker into the cardboard box". It has no memory and no planning: it only ever
executes the single instruction you give it, so decompose the goal yourself and
issue the next step when the current one looks finished.

You are shown the robot's current camera views and joint state.

Reply with ONLY a JSON object:
  {"task": "<next instruction>", "done": false, "reasoning": "<one sentence>"}
Set "done": true and "task": null when the goal is achieved.
Keep the current instruction by setting "task": null while it is still in progress.
"""


class GeminiAgent:
    """Google Gemini as the high-level planner (e.g. a Robotics-ER model).

    The model id is NOT hardcoded — pass ``model=`` or set ``HARNESS_MODEL``.
    Picking a default here would mean guessing at a name that changes between
    releases and failing obscurely when it is wrong; failing immediately with
    "set HARNESS_MODEL" is more useful than a 404 from inside a control loop.

    Needs ``pip install google-genai`` and ``GOOGLE_API_KEY``. Neither is a
    dependency of this repo: the harness runs fully on ScriptedAgent, and only
    this class pulls the SDK in.
    """

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_images: int = 3):
        self.model = model or os.environ.get("HARNESS_MODEL") or ""
        if not self.model:
            raise RuntimeError(
                "GeminiAgent needs a model id. Set HARNESS_MODEL=<model> or pass "
                "model=. This is deliberately not defaulted — see the class docstring.")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GeminiAgent needs GOOGLE_API_KEY in the environment.")
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. `pip install google-genai`, or run "
                "the harness with --agent=scripted, which needs no SDK.") from e
        self._client = genai.Client(api_key=self.api_key)
        self.max_images = max_images

    def decide(self, goal, frames, state, history) -> Decision:
        parts: list = [f"GOAL: {goal}"]
        if history:
            recent = "\n".join(
                f"- {h.task or '(unchanged)'}: {h.reasoning}" for h in history[-5:])
            parts.append(f"RECENT STEPS YOU ISSUED:\n{recent}")
        parts.append(f"JOINT STATE: {json.dumps(state)[:800]}")
        # Cap the image count: every frame is tokens and latency, and the third
        # view rarely changes a decision the two wrist views did not already make.
        for name, blob in list(frames.items())[: self.max_images]:
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": base64.b64encode(blob).decode()}})
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=parts,
                config={"system_instruction": _SYSTEM,
                        "response_mime_type": "application/json"},
            )
            data = json.loads((resp.text or "{}").strip())
        except Exception as e:
            # An agent failure must be visible and non-fatal: the loop keeps the
            # current task rather than the robot stopping or, worse, being sent
            # somewhere arbitrary by a half-parsed reply.
            logger.exception("agent call failed")
            return Decision(error=f"{type(e).__name__}: {e}",
                            reasoning="call failed — keeping current task")
        task = data.get("task")
        return Decision(task=task if isinstance(task, str) and task.strip() else None,
                        done=bool(data.get("done")),
                        reasoning=str(data.get("reasoning", ""))[:400])


def build(kind: str, **kw):
    """Factory used by the server's --agent flag."""
    if kind == "scripted":
        return ScriptedAgent(**kw)
    if kind == "gemini":
        return GeminiAgent(**kw)
    raise SystemExit(f"unknown agent {kind!r} (scripted|gemini)")
