"""OpenAI-compatible Chat transport for the public selector interface."""

from openai import OpenAI


class GatewayClient(OpenAI):
    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
