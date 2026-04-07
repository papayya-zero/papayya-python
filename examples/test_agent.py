"""Test agent with a simple tool for end-to-end validation."""

from papayya import Agent, tool


@tool
def get_company_info(company_name: str) -> str:
    """Look up basic information about a company."""
    # Simulated tool — in production this would call a real API
    data = {
        "stripe": "Stripe is a financial infrastructure platform. Founded 2010. HQ: San Francisco. Revenue: ~$14B (2023).",
        "anthropic": "Anthropic is an AI safety company. Founded 2021. HQ: San Francisco. Known for Claude AI models.",
        "openai": "OpenAI is an AI research company. Founded 2015. HQ: San Francisco. Known for GPT models and ChatGPT.",
    }
    key = company_name.lower().strip()
    return data.get(key, f"No data found for '{company_name}'.")


@tool
def summarize(text: str) -> str:
    """Create a concise summary of the given text."""
    # Simulated — just returns the text truncated
    if len(text) > 200:
        return text[:200] + "..."
    return text


agent = Agent(
    name="research-agent",
    model="gpt-4o-mini",
    instructions="You are a research assistant. When asked about a company, use the get_company_info tool to look it up, then summarize the findings.",
    tools=[get_company_info, summarize],
    max_steps=10,
    budget_usd=1.00,
)
