from decisions import AsyncClient, Client, Predicate, RefusalAnswer
from decisions.providers import Jeff, OpenAI


def invalid_usage(client: Client, async_client: AsyncClient, provider: OpenAI) -> None:
    client.predicate(123)  # type: ignore[arg-type]
    client.predicate("Ready?", state=123)  # type: ignore[arg-type]
    client.predicate("Ready?", images=[b"bytes"])  # type: ignore[list-item]
    client.choice("Team?", choices=[123])  # type: ignore[arg-type]
    client.score("Urgent?", levels=[True])  # type: ignore[list-item]
    client.decide({"ready": "Ready?"})  # type: ignore[dict-item]
    Predicate(123)  # type: ignore[arg-type]
    OpenAI(api_key=123)  # type: ignore[arg-type]
    Jeff()  # type: ignore[call-arg]
    provider.model = "changed"  # type: ignore[misc]
    provider.api_key = "changed"  # type: ignore[misc]
    provider.base_url = "http://other.test"  # type: ignore[misc]
    client.provider = Jeff("http://other.test")  # type: ignore[misc]
    async_client.provider = provider  # type: ignore[misc]
    answer = client.predicate("Ready?")
    answer.probability  # type: ignore[union-attr]
    if isinstance(answer, RefusalAnswer):
        answer.probability  # type: ignore[attr-defined]


async def invalid_async_usage(client: AsyncClient) -> None:
    await client.predicate(123)  # type: ignore[arg-type]
    await client.predicate("Ready?", state=123)  # type: ignore[arg-type]
    await client.predicate("Ready?", images=[b"bytes"])  # type: ignore[list-item]
    await client.choice("Team?", choices=[123])  # type: ignore[arg-type]
    await client.score("Urgent?", levels=[True])  # type: ignore[list-item]
    await client.decide({"ready": "Ready?"})  # type: ignore[dict-item]
    answer = await client.predicate("Ready?")
    answer.probability  # type: ignore[union-attr]
