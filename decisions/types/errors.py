"""Operational decisions failures"""


class DecisionsError(Exception):
    """Base class for decisions failures"""


class ConnectionError(DecisionsError):
    """A provider request failed before a response could be used"""


class TimeoutError(ConnectionError):
    """A provider request exceeded a transport timeout"""


class HTTPError(DecisionsError):
    """Provider returned an unsuccessful HTTP response"""

    def __init__(self, status: int, provider: str, req_id: str | None = None) -> None:
        self.status = status
        self.provider = provider
        self.req_id = req_id
        super().__init__(f"{provider} returned HTTP {status}")


class InvalidResponseError(DecisionsError):
    """A provider response could not be decoded or did not match its schema"""
