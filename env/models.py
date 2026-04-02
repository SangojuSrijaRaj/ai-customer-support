from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class UrgencyLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Department(str, Enum):
    billing = "billing"
    technical = "technical"
    general = "general"
    escalation = "escalation"
    sales = "sales"


class EmailObservation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    attachments: List[str] = Field(default_factory=list)
    task_id: str
    task_description: str


class AgentAction(BaseModel):
    urgency: UrgencyLevel
    department: Department
    reply_draft: Optional[str] = None
    escalate: bool = False
    tags: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: AgentAction


class StateResponse(BaseModel):
    current_observation: Optional[EmailObservation] = None
    current_task_id: Optional[str] = None
    step_count: int = 0
    total_reward: float = 0.0
    done: bool = True


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    score_range: List[float]


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
