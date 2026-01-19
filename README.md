# DataUp Python SDK

Official Python SDK for the DataUp API.

## Installation

### Core SDK

```bash
pip install dataup
```

### With Optional Dependencies

For evaluation providers, install with extras:

```bash
# Ultralytics provider
pip install dataup[ultralytics]

# Roboflow provider
pip install dataup[roboflow]

# All providers
pip install dataup[all]
```

See [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md) for more details.

## Quick Start

```python
from dataup import DataUpClient

# Initialize the client with your API key
client = DataUpClient(api_key="your_key_id.your_key_secret")

# List all agents
agents = client.agents.list()
for agent in agents.items:
    print(f"{agent.id}: {agent.name}")

# Close the client when done
client.close()
```

## Features

- **Sync and Async clients** - Choose the client that fits your use case
- **Type-safe** - Full type hints and Pydantic models
- **Pagination helpers** - Easy iteration through paginated results
- **Comprehensive error handling** - Custom exceptions for different error types

## Usage

### Context Manager

```python
from dataup import DataUpClient

with DataUpClient(api_key="your_key_id.your_key_secret") as client:
    agent = client.agents.get("agent_id")
    print(agent.name)
```

### Async Client

```python
import asyncio
from dataup import AsyncDataUpClient

async def main():
    async with AsyncDataUpClient(api_key="your_key_id.your_key_secret") as client:
        agents = await client.agents.list()
        for agent in agents.items:
            print(agent.name)

asyncio.run(main())
```

### Creating an Agent

```python
from dataup import DataUpClient, AgentCreate, AgentProvider, AgentType

client = DataUpClient(api_key="...")

agent = client.agents.create(AgentCreate(
    name="My Object Detector",
    endpoint="https://my-model.example.com/predict",
    auth_token="my-auth-token",
    provider=AgentProvider.CUSTOM,
    agent_type=AgentType.DETECTOR,
))
print(f"Created agent: {agent.id}")
```

### Running Inference

```python
from dataup import DataUpClient, InferenceRequest, DetectorParams

client = DataUpClient(api_key="...")

response = client.agents.infer(
    "agent_id",
    InferenceRequest(
        image_urls=["https://example.com/image.jpg"],
        params=DetectorParams(threshold=0.7),
    )
)

for result in response.data:
    print(f"Image: {result.image_id}")
    for label in result.labels:
        print(f"  - {label.label}: {label.score:.2f}")
```

### Pagination

Iterate through all pages automatically:

```python
from dataup import DataUpClient, paginate

client = DataUpClient(api_key="...")

# Iterate through all agents
for agent in paginate(client.agents.list, provider="ultralytics"):
    print(f"Agent: {agent.name}")
```

Or handle pagination manually:

```python
page = client.agents.list(size=20)
while True:
    for agent in page.items:
        print(agent.name)
    if not page.has_next():
        break
    page = client.agents.list(cursor=page.cursor, size=20)
```

### Error Handling

```python
from dataup import (
    DataUpClient,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
    RateLimitError,
)

client = DataUpClient(api_key="...")

try:
    agent = client.agents.get("nonexistent-id")
except NotFoundError as e:
    print(f"Agent not found: {e}")
except PermissionDeniedError as e:
    print(f"Access denied: {e}")
except ValidationError as e:
    print(f"Invalid request: {e}")
except RateLimitError as e:
    print(f"Rate limited: {e}")
```

### Evaluations

```python
from dataup import DataUpClient, EvaluationRunner, UltralyticsProvider
from dataup.cvat import CVATClient

# Set up clients
dataup = DataUpClient(api_key="...")
cvat = CVATClient(api_token="...")

# Set up inference provider
provider = UltralyticsProvider()
provider.load_model("yolov8n.pt")

# Run evaluation
runner = EvaluationRunner(dataup, cvat, provider)
result = runner.run_and_submit(
    task_id=123,
    agent_name="my-model",
    agent_version="1.0.0"
)

# Access metrics
print(f"mAP: {result.summary_metrics.ap:.3f}")
print(f"mAP@50: {result.summary_metrics.ap50:.3f}")
```

### Evaluation Providers

Use inference providers for local model evaluation:

```python
from dataup import UltralyticsProvider

# Requires: pip install dataup[ultralytics]
provider = UltralyticsProvider()
provider.load_model("yolov8n.pt")
labels = provider.predict(image, conf=0.25)
```

See [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md) for installation details.

## API Reference

### DataUpClient

| Method | Description |
|--------|-------------|
| `agents.list()` | List agents with optional filters |
| `agents.get(id)` | Get a specific agent |
| `agents.create(agent)` | Create a new agent |
| `agents.update(id, agent)` | Update an agent |
| `agents.delete(id)` | Delete an agent |
| `agents.infer(id, request)` | Run inference |
| `agents.activate(id)` | Activate an agent |
| `agents.deactivate(id)` | Deactivate an agent |
| `agents.get_monthly_usage(id)` | Get usage statistics |
| `evaluations.list()` | List evaluations |
| `evaluations.get(id)` | Get evaluation with metrics |
| `evaluations.create(evaluation)` | Create a new evaluation |
| `evaluations.delete(id)` | Delete an evaluation |
| `evaluations.ingest_batch(id, batch)` | Ingest frame batch |
| `evaluations.finalize(id)` | Finalize and compute metrics |
| `evaluations.get_frames(id)` | Get evaluation frames |
| `evaluations.get_job_metrics(id)` | Get job-level metrics |

## Configuration

```python
from dataup import DataUpClient

client = DataUpClient(
    api_key="your_key_id.your_key_secret",
    base_url="https://api.data-up.io",  # Default
    timeout=30.0,  # Request timeout in seconds
)
```

## License

MIT
