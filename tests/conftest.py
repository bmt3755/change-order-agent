import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any agent modules are imported so LangSmith tracing
# and real API keys are available for both smoke tests and e2e tests.
load_dotenv(Path(__file__).parent.parent / ".env")

# Smoke tests never make real API calls — fall back to dummy key so the
# OpenAI client doesn't raise "Missing credentials" at module load time.
os.environ.setdefault("OPENAI_API_KEY", "sk-smoke-test-dummy")
