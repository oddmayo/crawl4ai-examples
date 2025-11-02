# imports

import nest_asyncio, os, asyncio, json
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, 
    LLMConfig, LLMExtractionStrategy
)

# Reusable llm crawling function

async def extract_with_llm(
    url: str,
    fields: List[str],                        # e.g. ["name", "price"] or ["summary"]
    provider: str = "ollama/qwen2.5:3b",      # switch models and providers
    api_token: Optional[str] = None,          # for non-local providers if needed
    instruction: Optional[str] = None,        # site-specific prompt
    max_tokens: int = 500,                    # for more difficult websites
    temperature: float = 0.0,                 # 0 = deterministic
    input_format: str = "markdown",           # or "html"
    apply_chunking: bool = False,             # off unless pages are huge
    chunk_token_threshold: int = 1000,        # only used if chunking
    overlap_rate: float = 0.0,                # only used if chunking
    headless: bool = True,                    # browser settings
    text_mode: bool = True,
    light_mode: bool = True,
    cache_mode: CacheMode = CacheMode.BYPASS  # bypass cache by default
) -> Any:
    """
    Extracts specific fields from a webpage using Crawl4AI's LLM extraction strategy,
    without relying on Pydantic. Produces consistent output across all environments.
    """

    # --- Build a minimal JSON schema manually ---
    schema = {
        "type": "object",
        "properties": {f: {"type": "string"} for f in fields},
        "required": fields,
    }

    # --- Default LLM instruction ---
    if instruction is None:
        example = "{" + ", ".join([f'"{f}": "example {f}"' for f in fields]) + "}"
        instruction = (
            "From the page content, extract these fields in strict JSON with exactly these keys: "
            f"{fields}. Return a JSON object like: {example}"
        )

    # --- Define the extraction strategy ---
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider=provider, api_token=api_token),
        schema=schema,
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=chunk_token_threshold,
        overlap_rate=overlap_rate,
        apply_chunking=apply_chunking,
        input_format=input_format,
        extra_args={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )

    # --- Configure crawler ---
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=cache_mode,
    )

    browser_cfg = BrowserConfig(
        headless=headless,
        text_mode=text_mode,
        light_mode=light_mode,
    )

    # --- Run crawl and extract data ---
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)

        if not result.success:
            raise RuntimeError(result.error_message)

        # Return parsed JSON
        try:
            return json.loads(result.extracted_content)
        except Exception:
            # Return raw content if not valid JSON
            return result.extracted_content
