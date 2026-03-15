"""Split web tool implementation package."""

from unclaw.tools.web.constants import FETCH_URL_TEXT_DEFINITION, SEARCH_WEB_DEFINITION
from unclaw.tools.web.entrypoints import fetch_url_text, register_web_tools, search_web

__all__ = [
    "FETCH_URL_TEXT_DEFINITION",
    "SEARCH_WEB_DEFINITION",
    "fetch_url_text",
    "register_web_tools",
    "search_web",
]
