"""Shared text, URL, and noise-filtering helpers for the web stack."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse, urlunparse

from unclaw.tools.web.constants import MAX_SUMMARY_POINT_CHARS

LOW_VALUE_RESULT_TITLES = {"accueil", "home", "homepage", "index"}
ARTICLE_PATH_CUES = frozenset(
    {
        "analysis",
        "article",
        "articles",
        "blog",
        "blogs",
        "entry",
        "feature",
        "features",
        "post",
        "posts",
        "recap",
        "report",
        "reports",
        "story",
        "stories",
        "update",
        "updates",
    }
)
LIVE_STREAMING_PATH_CUES = frozenset(
    {
        "direct",
        "directs",
        "emission",
        "emissions",
        "en-direct",
        "live",
        "player",
        "programme",
        "programmes",
        "regarder",
        "replay",
        "stream",
        "streaming",
        "tv",
        "watch",
    }
)
HUB_PATH_CUES = frozenset(
    {
        "archive",
        "archives",
        "category",
        "categories",
        "index",
        "latest",
        "listing",
        "live",
        "page",
        "section",
        "tag",
        "tags",
        "topics",
        "updates",
    }
)
LOW_VALUE_PATH_CUES = frozenset(
    {
        "about",
        "account",
        "contact",
        "donate",
        "help",
        "join",
        "legal",
        "login",
        "logout",
        "privacy",
        "register",
        "settings",
        "share",
        "signin",
        "signup",
        "subscribe",
        "support",
        "terms",
    }
)
GENERIC_LINK_TEXTS = frozenset(
    {
        "continue reading",
        "home",
        "learn more",
        "menu",
        "more",
        "next",
        "older posts",
        "previous",
        "read more",
        "see more",
        "view more",
    }
)
MATCH_BOILERPLATE_PREFIXES = (
    "all rights reserved",
    "cookie ",
    "copyright ",
    "menu ",
    "sign in",
    "skip to",
)
NOISE_SIGNAL_PHRASES = frozenset(
    {
        "accept cookies",
        "accepter les cookies",
        "acceder aux notifications",
        "account settings",
        "all rights reserved",
        "already a subscriber",
        "article reserve",
        "conditions generales",
        "consentement",
        "contenu reserve",
        "cookie policy",
        "cookie preferences",
        "creer un compte",
        "data protection",
        "deja abonne",
        "deja inscrit",
        "donnees personnelles",
        "en poursuivant",
        "en savoir plus et gerer",
        "gerer les cookies",
        "gerer mes preferences",
        "inscription gratuite",
        "log in to",
        "manage preferences",
        "manage your subscription",
        "mon compte",
        "newsletter signup",
        "nos partenaires",
        "notre politique",
        "nous utilisons des cookies",
        "offre numerique",
        "offre speciale",
        "parametres de confidentialite",
        "partenaires data",
        "politique de confidentialite",
        "privacy policy",
        "privacy settings",
        "profitez de",
        "se connecter",
        "sign in to",
        "sign up for",
        "subscribe to",
        "subscription required",
        "terms of service",
        "tous droits reserves",
        "use of cookies",
        "utilisation de cookies",
        "votre abonnement",
        "votre consentement",
        "your privacy",
        "your subscription",
    }
)
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "au",
        "aujourdhui",
        "aujourd",
        "ce",
        "ces",
        "cette",
        "d",
        "de",
        "des",
        "du",
        "en",
        "est",
        "et",
        "for",
        "hui",
        "il",
        "is",
        "la",
        "l",
        "le",
        "les",
        "of",
        "on",
        "ou",
        "pour",
        "q",
        "qu",
        "que",
        "quel",
        "quelle",
        "quelles",
        "quels",
        "s",
        "the",
        "to",
        "today",
        "un",
        "une",
        "what",
    }
)
COPULAR_TOKENS = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "est",
        "etaient",
        "etait",
        "furent",
        "is",
        "sera",
        "seront",
        "sont",
        "was",
        "were",
    }
)
LOW_VALUE_EXTENSIONS = (
    ".css",
    ".ico",
    ".jpg",
    ".jpeg",
    ".js",
    ".json",
    ".pdf",
    ".png",
    ".rss",
    ".svg",
    ".xml",
    ".zip",
)

_SITE_DESCRIPTIVE_CUES = (
    "retrouvez toute l",
    "retrouvez toutes les",
    "suivez l actualite",
    "toute l actualite",
    "toutes les actualites",
    "toute l information",
    "decouvrez les dernieres",
    "decouvrez toute l",
    "decouvrez toutes les",
    "bienvenue sur",
    "welcome to our",
    "visit our",
    "follow us on",
    "suivez nous sur",
    "retrouvez nous sur",
    "abonnez vous",
    "stay up to date",
    "stay informed",
    "suivez en direct",
    "suivez en temps reel",
    "regardez en direct",
    "a suivre en direct",
    "en direct sur",
    "en continu sur",
    "your source for",
    "your daily source",
    "votre source d",
    "tout savoir sur",
    "l essentiel de l actualite",
    "toute l info",
)
_PROMOTIONAL_CUES = (
    "abonnez vous",
    "commencez votre",
    "creez votre compte",
    "decouvrez nos offres",
    "essai gratuit",
    "essayez gratuitement",
    "free trial",
    "get started",
    "inscrivez vous",
    "join now",
    "offre d abonnement",
    "offre speciale",
    "profitez de notre",
    "sign up today",
    "start your",
    "subscribe now",
    "try for free",
    "upgrade your",
)
_PROMPT_CONTROL_LINE_PATTERN = re.compile(
    r"^\s*(?:[>*-]\s*)?(?:system|user|assistant|tool|developer)\s*:\s*",
    flags=re.IGNORECASE,
)
_PROMPT_CONTROL_PHRASE_PATTERN = re.compile(
    r"\b(?:ignore|disregard)\s+(?:all\s+)?previous instructions\b|"
    r"\bact as\b|"
    r"\byou are now\b|"
    r"\bdeveloper message\b",
    flags=re.IGNORECASE,
)


def sanitize_model_visible_text(text: str) -> str:
    """Strip obvious control-text prompt injections before model-visible use."""
    sanitized_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        normalized_line = " ".join(raw_line.split())
        if not normalized_line:
            if sanitized_lines and not previous_blank:
                sanitized_lines.append("")
            previous_blank = True
            continue

        sanitized_line = _sanitize_instruction_like_line(normalized_line)
        if not sanitized_line:
            continue

        sanitized_lines.append(sanitized_line)
        previous_blank = False

    return "\n".join(sanitized_lines).strip()


def normalize_text(text: str) -> str:
    normalized_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue

        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


def fold_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def text_tokens(text: str) -> tuple[str, ...]:
    return tuple(fold_for_match(text).split())


def content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in text_tokens(text)
        if token not in QUERY_STOPWORDS and (len(token) > 2 or token.isdigit())
    )


def keyword_overlap_score(text: str, keywords: tuple[str, ...]) -> int:
    if not keywords:
        return 0
    text_token_set = set(text.split())
    return sum(1 for keyword in keywords if keyword in text_token_set)


def iter_passages(text: str) -> tuple[str, ...]:
    passages = tuple(
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    )
    if passages:
        return passages
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def is_informative_passage(text: str, *, title: str) -> bool:
    words = text.split()
    if len(words) < 8 or looks_like_title_echo(text, title):
        return False

    lowered = fold_for_match(text)
    if lowered.startswith(MATCH_BOILERPLATE_PREFIXES):
        return False
    if passage_has_noise_signals(text):
        return False
    if looks_site_descriptive(text):
        return False
    if looks_promotional(text):
        return False
    return True


def looks_low_value_page(
    *,
    url: str,
    text: str,
    title: str,
    link_count: int,
) -> bool:
    if not text or text == "[empty response body]":
        return True
    token_count = len(text_tokens(text))
    if token_count < 18:
        return True
    if looks_like_title_echo(text, title):
        return True
    if url_looks_homepage_like(url) and token_count < 80:
        return True
    if link_count >= 8 and token_count < 120:
        return True
    if link_count >= 12 and token_count < 200:
        return True
    if passage_has_noise_signals(text) and token_count < 100:
        return True
    return False


def looks_like_title_echo(text: str, title: str) -> bool:
    title_token_list = text_tokens(title)
    body_tokens = text_tokens(text)
    if not title_token_list or not body_tokens:
        return False

    short_text = len(body_tokens) <= max(len(title_token_list) + 6, 18)
    if not short_text:
        return False

    title_signature = " ".join(title_token_list)
    text_signature = " ".join(body_tokens[: len(title_token_list) + 6])
    if text_signature.startswith(title_signature):
        return True

    title_token_set = set(title_token_list)
    overlap = sum(1 for token in body_tokens if token in title_token_set)
    return overlap >= max(3, len(title_token_list) - 1)


def looks_boilerplate_text(text: str) -> bool:
    return fold_for_match(text).startswith(MATCH_BOILERPLATE_PREFIXES)


def looks_site_descriptive(text: str) -> bool:
    folded = fold_for_match(text)
    return any(cue in folded for cue in _SITE_DESCRIPTIVE_CUES)


def looks_promotional(text: str) -> bool:
    folded = fold_for_match(text)
    return any(cue in folded for cue in _PROMOTIONAL_CUES)


def passage_has_noise_signals(text: str) -> bool:
    folded = fold_for_match(text)
    return any(phrase in folded for phrase in NOISE_SIGNAL_PHRASES)


def passage_noise_score(text: str) -> float:
    folded = fold_for_match(text)
    hits = sum(1 for phrase in NOISE_SIGNAL_PHRASES if phrase in folded)
    if hits == 0:
        return 0.0
    return min(hits * 3.0, 9.0)


def truncate_sentences(text: str, *, max_sentences: int, max_chars: int) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    if not sentences:
        return clip_text(text, limit=max_chars)

    selected = " ".join(sentences[:max_sentences]).strip()
    if selected:
        return clip_text(selected, limit=max_chars)
    return clip_text(text, limit=max_chars)


def join_summary_parts(parts: tuple[str, ...]) -> str:
    cleaned_parts = [part.strip() for part in parts if part.strip()]
    if not cleaned_parts:
        return ""
    joined = ". ".join(strip_terminal_punctuation(part) for part in cleaned_parts)
    if joined and joined[-1] not in ".!?":
        joined += "."
    return joined


def strip_terminal_punctuation(text: str) -> str:
    return text.strip().rstrip(" ,;:.!?")


def clip_summary_text(value: str, *, limit: int = MAX_SUMMARY_POINT_CHARS) -> str:
    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized

    for separator in (". ", "; ", ", "):
        boundary = normalized.rfind(separator, 0, limit)
        if boundary >= int(limit * 0.65):
            clipped = normalized[: boundary + (1 if separator == ". " else 0)].rstrip()
            if clipped and clipped[-1] not in ".!?":
                clipped += "."
            return clipped

    return clip_text(normalized, limit=limit)


def merge_unique_strings(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in (*left, *right):
        key = value.casefold()
        if key in seen:
            continue
        merged.append(value)
        seen.add(key)
    return tuple(merged)


def clip_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized

    clipped = normalized[: limit - 3].rstrip(" ,;:.")
    return f"{clipped}..."


def is_supported_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def canonicalize_url(url: str) -> str:
    if not is_supported_url(url):
        return ""

    parsed = urlparse(url)
    normalized_path = parsed.path or "/"
    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            normalized_path,
            "",
            parsed.query,
            "",
        )
    )


def registered_domain(hostname: str) -> str:
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname.startswith("www."):
        normalized_hostname = normalized_hostname[4:]
    parts = normalized_hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return normalized_hostname


def url_path_segments(url: str) -> tuple[str, ...]:
    parsed = urlparse(url)
    return tuple(segment for segment in parsed.path.split("/") if segment)


def url_looks_homepage_like(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        return True
    return path.casefold() in {"/accueil", "/home", "/index.html"}


def url_looks_article_like(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path
    path_segments = url_path_segments(url)
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", path):
        return True
    if re.search(r"/20\d{2}-\d{2}-\d{2}", path):
        return True
    if any(segment.casefold() in ARTICLE_PATH_CUES for segment in path_segments):
        return True
    if path_segments and path_segments[-1].count("-") >= 3:
        return True
    return False


def url_looks_archive_like(url: str) -> bool:
    path_segments = {segment.casefold() for segment in url_path_segments(url)}
    return bool(path_segments & HUB_PATH_CUES)


def url_looks_live_or_streaming(url: str) -> bool:
    path_segments = {segment.casefold() for segment in url_path_segments(url)}
    return bool(path_segments & LIVE_STREAMING_PATH_CUES)


def url_looks_low_value(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.fragment:
        return True
    path = parsed.path.casefold()
    if any(path.endswith(extension) for extension in LOW_VALUE_EXTENSIONS):
        return True
    path_segments = {segment.casefold() for segment in url_path_segments(url)}
    if path_segments & LOW_VALUE_PATH_CUES:
        return True
    return False


def looks_generic_result_title(*, title: str, hostname: str) -> bool:
    folded_title = fold_for_match(title)
    if folded_title in LOW_VALUE_RESULT_TITLES:
        return True

    hostname_tokens = [
        token
        for token in re.split(r"[.\-]+", hostname.casefold())
        if token and token not in {"com", "fr", "net", "org", "www"}
    ]
    if hostname_tokens and folded_title == " ".join(hostname_tokens):
        return True
    return False


def link_text_looks_generic(text: str) -> bool:
    return fold_for_match(text) in GENERIC_LINK_TEXTS


def _sanitize_instruction_like_line(line: str) -> str:
    if _PROMPT_CONTROL_LINE_PATTERN.match(line):
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", line)
        if sentence.strip()
    ]
    kept_sentences = [
        sentence
        for sentence in sentences
        if not _looks_instruction_like_sentence(sentence)
    ]
    if kept_sentences:
        return " ".join(kept_sentences).strip()

    if _looks_instruction_like_sentence(line):
        return ""
    return line.strip()


def _looks_instruction_like_sentence(text: str) -> bool:
    return bool(
        _PROMPT_CONTROL_LINE_PATTERN.match(text)
        or _PROMPT_CONTROL_PHRASE_PATTERN.search(text)
    )


__all__ = [
    "COPULAR_TOKENS",
    "QUERY_STOPWORDS",
    "canonicalize_url",
    "clip_summary_text",
    "clip_text",
    "content_tokens",
    "fold_for_match",
    "is_informative_passage",
    "is_supported_url",
    "iter_passages",
    "join_summary_parts",
    "keyword_overlap_score",
    "link_text_looks_generic",
    "looks_boilerplate_text",
    "looks_generic_result_title",
    "looks_like_title_echo",
    "looks_low_value_page",
    "looks_promotional",
    "looks_site_descriptive",
    "merge_unique_strings",
    "normalize_text",
    "passage_has_noise_signals",
    "passage_noise_score",
    "registered_domain",
    "sanitize_model_visible_text",
    "strip_terminal_punctuation",
    "text_tokens",
    "truncate_sentences",
    "url_looks_archive_like",
    "url_looks_article_like",
    "url_looks_homepage_like",
    "url_looks_live_or_streaming",
    "url_looks_low_value",
    "url_path_segments",
]
