"""Operational decisions failures"""


class DecisionsError(Exception):
    """Base class for decisions failures"""


class HTTPError(DecisionsError):
    """Provider returned an unsuccessful HTTP response"""

    def __init__(self, *, status: int, provider: str, req_id: str | None = None) -> None:
        self.status = status
        self.provider = provider
        self.req_id = req_id
        super().__init__(f"{provider} returned HTTP {status}")
