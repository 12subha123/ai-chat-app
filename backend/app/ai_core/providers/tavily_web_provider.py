from typing import Any

import requests

from ..external_provider import ExternalProvider
from ..external_provider_config import ExternalProviderConfig


class TavilyWebProvider(ExternalProvider):
    """
    Real Tavily web-search provider.

    Automatically chooses a news-focused search for
    questions asking about current or recent events.
    """

    def __init__(self):
        super().__init__(
            config=ExternalProviderConfig(
                name="tavily",
                api_key_env="TAVILY_API_KEY",
                base_url_env="TAVILY_API_BASE_URL",
            )
        )

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def available(self) -> bool:
        return self.config.configured

    @staticmethod
    def _is_recent_query(query: str) -> bool:
        recent_keywords = (
            "latest",
            "today",
            "currently",
            "current",
            "recent",
            "recently",
            "news",
            "this week",
            "this month",
            "breaking",
            "just announced",
            "new development",
        )

        normalized_query = query.lower()

        return any(
            keyword in normalized_query
            for keyword in recent_keywords
        )

    def execute(self, request: Any) -> dict:
        if not self.available:
            return {
                "status": "not_connected",
                "provider": self.name,
                "message": "Tavily API key is not configured.",
            }

        if not hasattr(request, "message"):
            raise ValueError(
                "TavilyWebProvider requires a request with a message."
            )

        query = request.message.strip()

        if not query:
            raise ValueError(
                "Tavily search query cannot be empty."
            )

        api_key = self.config.get_api_key()

        base_url = self.config.get_base_url()

        if not base_url:
            base_url = "https://api.tavily.com"

        endpoint = f"{base_url.rstrip('/')}/search"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        }

        if self._is_recent_query(query):
            payload["topic"] = "news"
            payload["days"] = 7

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()