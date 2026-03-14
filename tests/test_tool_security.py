from __future__ import annotations

import shutil
import socket
from pathlib import Path
from urllib.error import URLError

import yaml

from unclaw.core.executor import ToolExecutor
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall


class _FakeHeaders:
    def __init__(
        self,
        *,
        content_type: str = "text/plain",
        charset: str = "utf-8",
    ) -> None:
        self._content_type = content_type
        self._charset = charset

    def get_content_type(self) -> str:
        return self._content_type

    def get_content_charset(self) -> str:
        return self._charset


class _FakeResponse:
    def __init__(
        self,
        *,
        url: str,
        body: str,
        content_type: str = "text/plain",
    ) -> None:
        self._url = url
        self._body = body.encode("utf-8")
        self.headers = _FakeHeaders(content_type=content_type)
        self.status = 200

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self._body
        return self._body[:size]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:  # type: ignore[no-untyped-def]
        del exc_type, exc, traceback
        return False


def _raise_url_timeout(
    request,
    timeout_seconds,
    allow_private_networks,
):  # type: ignore[no-untyped-def]
    del request, timeout_seconds, allow_private_networks
    raise URLError("timed out")


def test_read_tool_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret\n", encoding="utf-8")
    executor = ToolExecutor.with_default_tools(settings)

    result = executor.execute(
        ToolCall(
            tool_name="read_text_file",
            arguments={"path": str(outside_file)},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error


def test_list_tool_rejects_directories_outside_allowed_roots(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    outside_directory = tmp_path / "outside-dir"
    outside_directory.mkdir()
    executor = ToolExecutor.with_default_tools(settings)

    result = executor.execute(
        ToolCall(
            tool_name="list_directory",
            arguments={"path": str(outside_directory)},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error


def test_fetch_tool_blocks_private_network_targets_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        ],
    )

    result = executor.execute(
        ToolCall(
            tool_name="fetch_url_text",
            arguments={"url": "http://blocked.test"},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "blocked because" in result.error


def test_fetch_tool_allows_public_urls(monkeypatch, tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        lambda request, timeout_seconds, allow_private_networks: _FakeResponse(
            url=request.full_url,
            body="public content",
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="fetch_url_text",
            arguments={"url": "https://example.com/docs"},
        )
    )

    assert result.success is True
    assert "public content" in result.output_text


def test_fetch_tool_can_opt_in_to_private_network_access(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["security"]["tools"]["fetch"]["allow_private_networks"] = True
    app_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        lambda request, timeout_seconds, allow_private_networks: _FakeResponse(
            url=request.full_url,
            body="private content",
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="fetch_url_text",
            arguments={"url": "http://127.0.0.1:11434/api/tags"},
        )
    )

    assert result.success is True
    assert "private content" in result.output_text


def test_search_tool_returns_compact_structured_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body="""
            <html>
              <body>
                <div class="result">
                  <a class="result__a" href="https://example.com/docs/local-first-agents">
                    Example Docs
                  </a>
                  <div class="result__snippet">
                    Local-first agent design notes and reliability guidance.
                  </div>
                </div>
                <div class="result">
                  <a class="result__a" href="https://blog.example.com/posts/tooling-grounding">
                    Agent Notes
                  </a>
                  <div class="result__snippet">
                    Practical observations about lightweight local AI tooling.
                  </div>
                </div>
              </body>
            </html>
            """,
            page_bodies={
                "https://example.com/docs/local-first-agents": """
                <html><body><main>
                <p>Local-first agents keep user data on the device and only sync what is needed.</p>
                <p>The guide focuses on reliability, offline behavior, and predictable tool access.</p>
                </main></body></html>
                """,
                "https://blog.example.com/posts/tooling-grounding": """
                <html><body><article>
                <p>Practical local AI agents work best when search, fetch, and file tools stay small and explicit.</p>
                <p>The article highlights grounded summaries instead of forcing users to inspect every URL manually.</p>
                </article></body></html>
                """,
            },
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "local first agents"},
        )
    )

    assert result.success is True
    assert "Search query: local first agents" in result.output_text
    assert "Sources considered: 2" in result.output_text
    assert "Sources fetched: 2 of 2 attempted" in result.output_text
    assert "Evidence kept:" in result.output_text
    assert "Summary:" in result.output_text
    assert "Sources:" in result.output_text
    assert "Example Docs" in result.output_text
    assert "URL: https://example.com/docs/local-first-agents" in result.output_text
    assert "Takeaway:" in result.output_text
    assert "Note:" not in result.output_text
    assert (
        "- Local-first agents keep user data on the device and only sync what is needed."
        in result.output_text
    )
    assert result.payload is not None
    assert result.payload["initial_result_count"] == 2
    assert result.payload["fetch_success_count"] == 2
    assert result.payload["fetch_attempt_count"] == 2
    assert result.payload["evidence_count"] >= 2


def test_search_tool_uses_iterative_second_level_exploration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)
    requested_urls: list[str] = []

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body="""
            <html>
              <body>
                <div class="result">
                  <a class="result__a" href="https://example.com/updates">
                    Example Updates
                  </a>
                  <div class="result__snippet">
                    Archive of recent updates and release notes.
                  </div>
                </div>
              </body>
            </html>
            """,
            page_bodies={
                "https://example.com/updates": """
                <html><body><main>
                <p>Recent release archive and incident index.</p>
                <a href="/updates/release-2026-03-14-notes">March 14 release notes</a>
                <a href="/updates/install-fix-2026-03-14">Install fix details</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
                </main></body></html>
                """,
                "https://example.com/updates/release-2026-03-14-notes": """
                <html><body><article>
                <p>The March 14 release cut failed installs by 30 percent and shortened cold-start setup time.</p>
                <p>Operators also added a clearer retry path when a local model download stalls.</p>
                </article></body></html>
                """,
                "https://example.com/updates/install-fix-2026-03-14": """
                <html><body><article>
                <p>The install repair patch now resumes partially downloaded model files instead of starting over.</p>
                </article></body></html>
                """,
            },
            requested_urls=requested_urls,
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "latest install reliability release"},
        )
    )

    assert result.success is True
    assert "Sources considered: 3" in result.output_text
    assert (
        "The install repair patch now resumes partially downloaded model files instead of starting over."
        in result.output_text
        or "Operators also added a clearer retry path when a local model download stalls."
        in result.output_text
    )
    assert "https://example.com/updates/release-2026-03-14-notes" in requested_urls
    assert result.payload is not None
    assert result.payload["considered_candidate_count"] == 3
    assert result.payload["fetch_attempt_count"] == 3
    assert any(
        item["url"] == "https://example.com/updates/release-2026-03-14-notes"
        for item in result.payload["results"]
    )


def test_search_tool_respects_fetch_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)
    requested_urls: list[str] = []
    page_bodies: dict[str, str] = {}
    search_results: list[str] = []

    for index in range(1, 21):
        hub_url = f"https://example.com/archive/{index}"
        search_results.append(
            f"""
            <div class="result">
              <a class="result__a" href="{hub_url}">
                Archive {index}
              </a>
              <div class="result__snippet">
                Archive page {index} with recent release items.
              </div>
            </div>
            """
        )
        page_bodies[hub_url] = f"""
        <html><body><main>
        <p>Archive page {index} with the latest release entries.</p>
        <a href="/archive/{index}/release-1-2026-03-14">Release 1</a>
        <a href="/archive/{index}/release-2-2026-03-14">Release 2</a>
        <a href="/archive/{index}/release-3-2026-03-14">Release 3</a>
        <a href="/about">About</a>
        </main></body></html>
        """
        for child_index in range(1, 4):
            page_bodies[f"https://example.com/archive/{index}/release-{child_index}-2026-03-14"] = f"""
            <html><body><article>
            <p>Release {child_index} for archive {index} improved install recovery after interrupted downloads.</p>
            </article></body></html>
            """

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._should_stop_retrieval",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body=f"<html><body>{''.join(search_results)}</body></html>",
            page_bodies=page_bodies,
            requested_urls=requested_urls,
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "install recovery release archive"},
        )
    )

    page_fetches = [
        url for url in requested_urls if not url.startswith("https://html.duckduckgo.com/html/")
    ]

    assert result.success is True
    assert result.payload is not None
    assert result.payload["fetch_attempt_count"] == 30
    assert result.payload["fetch_success_count"] == 30
    assert len(page_fetches) == 30


def test_search_tool_respects_depth_cap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)
    requested_urls: list[str] = []

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._should_stop_retrieval",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body="""
            <html>
              <body>
                <div class="result">
                  <a class="result__a" href="https://example.com/archive">
                    Example Archive
                  </a>
                  <div class="result__snippet">
                    Release archive and monthly indexes.
                  </div>
                </div>
              </body>
            </html>
            """,
            page_bodies={
                "https://example.com/archive": """
                <html><body><main>
                <p>Monthly release archive index.</p>
                <a href="/archive/march-14">March 14 archive</a>
                <a href="/archive/february">February archive</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
                </main></body></html>
                """,
                "https://example.com/archive/march-14": """
                <html><body><main>
                <p>March 14 archive page with grouped article links.</p>
                <a href="/archive/march-14/details-2026-03-14">Detailed article</a>
                <a href="/archive/march-14/related-2026-03-14">Related article</a>
                <a href="/archive">Back to archive</a>
                <a href="/about">About</a>
                </main></body></html>
                """,
                "https://example.com/archive/february": """
                <html><body><main>
                <p>February archive index.</p>
                </main></body></html>
                """,
                "https://example.com/archive/march-14/details-2026-03-14": """
                <html><body><article>
                <p>This page should never be fetched because it sits beyond the depth cap.</p>
                </article></body></html>
                """,
                "https://example.com/archive/march-14/related-2026-03-14": """
                <html><body><article>
                <p>This sibling article should also stay unfetched at depth three.</p>
                </article></body></html>
                """,
            },
            requested_urls=requested_urls,
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "march 14 release archive"},
        )
    )

    assert result.success is True
    assert "https://example.com/archive" in requested_urls
    assert "https://example.com/archive/march-14" in requested_urls
    assert "https://example.com/archive/march-14/details-2026-03-14" not in requested_urls
    assert "https://example.com/archive/march-14/related-2026-03-14" not in requested_urls


def test_search_tool_deduplicates_evidence_across_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body="""
            <html>
              <body>
                <div class="result">
                  <a class="result__a" href="https://example.com/post-a">
                    Post A
                  </a>
                  <div class="result__snippet">
                    Install reliability recap.
                  </div>
                </div>
                <div class="result">
                  <a class="result__a" href="https://example.com/post-b">
                    Post B
                  </a>
                  <div class="result__snippet">
                    Another install reliability recap.
                  </div>
                </div>
              </body>
            </html>
            """,
            page_bodies={
                "https://example.com/post-a": """
                <html><body><article>
                <p>The release now resumes partially downloaded model files instead of restarting the transfer.</p>
                </article></body></html>
                """,
                "https://example.com/post-b": """
                <html><body><article>
                <p>The release now resumes partially downloaded model files instead of restarting the transfer.</p>
                </article></body></html>
                """,
            },
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "install reliability release"},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["evidence_count"] == 1
    assert len(result.payload["summary_points"]) == 1
    assert (
        result.payload["summary_points"][0]
        == "The release now resumes partially downloaded model files instead of restarting the transfer."
    )


def test_search_tool_prefers_article_like_child_pages_over_generic_parent_pages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body="""
            <html>
              <body>
                <div class="result">
                  <a class="result__a" href="https://example.com/releases">
                    Example Releases
                  </a>
                  <div class="result__snippet">
                    Release archive and update listings.
                  </div>
                </div>
              </body>
            </html>
            """,
            page_bodies={
                "https://example.com/releases": """
                <html><body><main>
                <p>Release listings and archive links.</p>
                <a href="/2026/03/14/major-release-notes">Major release notes</a>
                <a href="/releases/archive">Archive listing</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
                </main></body></html>
                """,
                "https://example.com/2026/03/14/major-release-notes": """
                <html><body><article>
                <p>The major release reduced failed installs and added clearer recovery steps for interrupted downloads.</p>
                <p>Startup after the first local warmup also became noticeably faster.</p>
                </article></body></html>
                """,
                "https://example.com/releases/archive": """
                <html><body><main>
                <p>Archive listing page.</p>
                </main></body></html>
                """,
            },
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "latest release install recovery"},
        )
    )

    assert result.success is True
    assert "1. Major release notes" in result.output_text
    assert "2. Example Releases" in result.output_text
    assert (
        result.output_text.index("1. Major release notes")
        < result.output_text.index("2. Example Releases")
    )
    assert result.payload is not None
    assert result.payload["results"][0]["url"] == "https://example.com/2026/03/14/major-release-notes"


def test_search_tool_summary_bullets_capture_findings_not_titles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body="""
            <html>
              <body>
                <div class="result">
                  <a class="result__a" href="https://example.com/update">
                    Quarterly Update
                  </a>
                  <div class="result__snippet">
                    Release coverage for the latest Example platform changes.
                  </div>
                </div>
                <div class="result">
                  <a class="result__a" href="https://community.example.com/recap">
                    Community Recap
                  </a>
                  <div class="result__snippet">
                    Users highlighted the most noticeable changes after upgrading.
                  </div>
                </div>
              </body>
            </html>
            """,
            page_bodies={
                "https://example.com/update": """
                <html><body><article>
                <p>Example Corp cut setup time by 30 percent and reduced failed installs in the latest release.</p>
                <p>The update also adds clearer recovery steps when local model downloads stall.</p>
                </article></body></html>
                """,
                "https://community.example.com/recap": """
                <html><body><article>
                <p>Users said the biggest improvement was faster startup after the first model warmup.</p>
                <p>Several posts also mentioned that search summaries now need fewer follow-up clicks.</p>
                </article></body></html>
                """,
            },
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "latest example release"},
        )
    )

    assert result.success is True
    assert (
        "- Example Corp cut setup time by 30 percent and reduced failed installs in the latest release."
        in result.output_text
    )
    assert (
        "- Users said the biggest improvement was faster startup after the first model warmup."
        in result.output_text
    )
    assert "- Quarterly Update" not in result.output_text
    assert "- Community Recap" not in result.output_text


def test_search_tool_handles_partial_read_failures_gracefully(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _build_search_open_request(
            search_body="""
            <html>
              <body>
                <div class="result">
                  <a class="result__a" href="https://example.com/news">
                    Example News
                  </a>
                  <div class="result__snippet">
                    Example Corp published a shipping update.
                  </div>
                </div>
                <div class="result">
                  <a class="result__a" href="https://blog.example.com/post">
                    Community Notes
                  </a>
                  <div class="result__snippet">
                    Users are discussing what changed in the latest release.
                  </div>
                </div>
              </body>
            </html>
            """,
            page_bodies={
                "https://example.com/news": """
                <html><body><article>
                <p>Example Corp says the latest release improves install reliability and startup speed.</p>
                </article></body></html>
                """,
            },
            page_errors={
                "https://blog.example.com/post": URLError("timed out"),
            },
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "latest example release"},
        )
    )

    assert result.success is True
    assert "Sources fetched: 1 of 2 attempted" in result.output_text
    assert (
        "- Example Corp says the latest release improves install reliability and startup speed."
        in result.output_text
    )
    assert (
        "- Users are discussing what changed in the latest release."
        in result.output_text
    )
    assert "Takeaway: Users are discussing what changed in the latest release." in result.output_text
    assert result.payload is not None
    assert result.payload["fetch_success_count"] == 1
    assert result.payload["fetch_attempt_count"] == 2


def test_search_tool_reports_provider_failures_cleanly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    monkeypatch.setattr(
        "unclaw.tools.web_tools.socket.getaddrinfo",
        lambda host, port, type=socket.SOCK_STREAM: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "unclaw.tools.web_tools._open_request",
        _raise_url_timeout,
    )

    result = executor.execute(
        ToolCall(
            tool_name="search_web",
            arguments={"query": "local first agents"},
        )
    )

    assert result.success is False
    assert result.error == "Could not search the web for 'local first agents': timed out"


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root


def _build_search_open_request(
    *,
    search_body: str,
    page_bodies: dict[str, str],
    page_errors: dict[str, Exception] | None = None,
    requested_urls: list[str] | None = None,
):
    resolved_page_errors = page_errors or {}

    def fake_open_request(
        request,
        timeout_seconds,
        allow_private_networks,
    ):  # type: ignore[no-untyped-def]
        del timeout_seconds, allow_private_networks
        if requested_urls is not None:
            requested_urls.append(request.full_url)
        if request.full_url.startswith("https://html.duckduckgo.com/html/"):
            return _FakeResponse(
                url=request.full_url,
                body=search_body,
                content_type="text/html",
            )
        if request.full_url in resolved_page_errors:
            raise resolved_page_errors[request.full_url]
        if request.full_url in page_bodies:
            return _FakeResponse(
                url=request.full_url,
                body=page_bodies[request.full_url],
                content_type="text/html",
            )
        raise AssertionError(f"Unexpected URL requested: {request.full_url}")

    return fake_open_request
