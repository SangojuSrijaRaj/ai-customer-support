import random
from typing import Optional
from env.models import (
    EmailObservation, AgentAction, StepResult,
    StateResponse, TaskInfo, TaskListResponse
)
from env.data import get_emails_for_task
from graders.graders import TASK_GRADERS

TASKS = [
    TaskInfo(id="task_easy", name="Basic Email Classification",
             description="Classify the urgency level and route the email to the correct department. No reply draft needed.",
             difficulty="easy", score_range=[0.0, 1.0]),
    TaskInfo(id="task_medium", name="Email Triage with Reply Draft",
             description="Classify urgency, route correctly, AND write a professional reply draft addressing the customer's issue.",
             difficulty="medium", score_range=[0.0, 1.0]),
    TaskInfo(id="task_hard", name="Multi-signal Escalation Decision",
             description="Classify urgency, route correctly, draft a reply, AND decide whether to escalate to a human agent.",
             difficulty="hard", score_range=[0.0, 1.0]),
]

TASK_IDS = [t.id for t in TASKS]


class EmailTriageEnv:
    def __init__(self):
        self._current_observation: Optional[EmailObservation] = None
        self._current_task_id: Optional[str] = None
        self._current_email_data: Optional[dict] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = True
        self._rng = random.Random()

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> EmailObservation:
        if seed is not None:
            self._rng.seed(seed)
        self._current_task_id = task_id if task_id in TASK_IDS else self._rng.choice(TASK_IDS)
        emails = get_emails_for_task(self._current_task_id)
        self._current_email_data = self._rng.choice(emails)
        task_info = next(t for t in TASKS if t.id == self._current_task_id)
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._current_observation = EmailObservation(
            email_id=self._current_email_data["email_id"],
            subject=self._current_email_data["subject"],
            body=self._current_email_data["body"],
            sender=self._current_email_data["sender"],
            timestamp=self._current_email_data["timestamp"],
            attachments=self._current_email_data.get("attachments", []),
            task_id=self._current_task_id,
            task_description=task_info.description,
        )
        return self._current_observation

    def step(self, action: AgentAction) -> StepResult:
        if self._done or self._current_email_data is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        grader = TASK_GRADERS[self._current_task_id]
        action_dict = {
            "urgency": action.urgency.value,
            "department": action.department.value,
            "reply_draft": action.reply_draft,
            "escalate": action.escalate,
            "tags": action.tags,
        }
        reward, grader_info = grader(action_dict, self._current_email_data["ground_truth"])
        self._total_reward += reward
        self._step_count += 1
        self._done = True
        return StepResult(
            observation=self._current_observation,
            reward=reward,
            done=self._done,
            info={"grader_details": grader_info, "step_count": self._step_count,
                  "total_reward": self._total_reward, "task_id": self._current_task_id},
        )

    def state(self) -> StateResponse:
        return StateResponse(
            current_observation=self._current_observation,
            current_task_id=self._current_task_id,
            step_count=self._step_count,
            total_reward=self._total_reward,
            done=self._done,
        )

    @staticmethod
    def list_tasks() -> TaskListResponse:
        return TaskListResponse(tasks=TASKS)
