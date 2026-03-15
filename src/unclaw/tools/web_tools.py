"""Public compatibility wrapper for the split web tool stack."""

from unclaw.tools.web import (
    FETCH_URL_TEXT_DEFINITION,
    SEARCH_WEB_DEFINITION,
    fetch_url_text,
    register_web_tools,
    search_web,
)

__all__ = [
    "FETCH_URL_TEXT_DEFINITION",
    "SEARCH_WEB_DEFINITION",
    "fetch_url_text",
    "search_web",
    "register_web_tools",
]
