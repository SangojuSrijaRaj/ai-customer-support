import sys
import os

# Ensure project root is on path so imports work when run as entry point
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.models import (
    ResetRequest, StepRequest, EmailObservation,
    StepResult, StateResponse, TaskListResponse
)
from env.environment import EmailTriageEnv

app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world customer support email triage environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnv()


@app.get("/")
def root():
    return {"message": "Email Triage OpenEnv is running.", "docs": "/docs", "health": "/health", "tasks": "/tasks"}


@app.get("/health")
def health():
    return {"status": "ok", "environment": "email-triage-env", "version": "1.0.0"}


@app.post("/reset", response_model=EmailObservation)
def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    obs = env.reset(task_id=request.task_id, seed=request.seed)
    return obs


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    try:
        result = env.step(request.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateResponse)
def state():
    return env.state()


@app.get("/tasks", response_model=TaskListResponse)
def tasks():
    return EmailTriageEnv.list_tasks()


def main():
    """
    Entry point for multi-mode deployment.
    Called via: project.scripts server = "server.app:main"
    """
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
