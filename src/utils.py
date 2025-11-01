# dependencies import

import nest_asyncio # for notebooks
import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai import LLMExtractionStrategy

# Reusable function

async def extract_with_llm(
    url: str,                                 # depending on wesbite
    fields: List[str],                        # e.g. ["name","description"] or ["summary"] or ["title","price"]
    provider: str = "ollama/qwen2.5:3b",      # switch models and providers
    api_token: Optional[str] = None,          # for non-local providers if needed
    instruction: Optional[str] = None,        # site-specific prompt
    max_tokens: int = 500,                    # for more difficult websites
    temperature: float = 0.0,                 # often 0 for extraction
    input_format: str = "markdown",           # or "html"
    apply_chunking: bool = False,             # off unless pages are huge
    chunk_token_threshold: int = 1000,        # only matters if chunking
    overlap_rate: float = 0.0,                # only matters if chunking
    headless: bool = True,                    # browser settings
    text_mode: bool = True,
    light_mode: bool = True,
    cache_mode: CacheMode = CacheMode.BYPASS  # bypass cache by default
) -> Any:
    # Build a dynamic schema from your field names (all required strings)
    Model = create_model("DynamicOutput", **{f: (str, ...) for f in fields})  # required strings

    # Default instruction tailored to the chosen fields if none provided
    if instruction is None:
        example = "{" + ", ".join([f'"{f}": "example {f}"' for f in fields]) + "}"
        instruction = (
            "From the crawled content, extract these fields in strict JSON with exactly these keys: "
            f"{fields}. Return a JSON object like: {example}"
        )

    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider=provider, api_token=api_token),
        schema=Model.model_json_schema(),
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=chunk_token_threshold,
        overlap_rate=overlap_rate,
        apply_chunking=apply_chunking,
        input_format=input_format,
        extra_args={"temperature": temperature, "max_tokens": max_tokens},
    )

    crawl_config = CrawlerRunConfig(extraction_strategy=llm_strategy, cache_mode=cache_mode)
    browser_cfg = BrowserConfig(headless=headless, text_mode=text_mode, light_mode=light_mode)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if not result.success:
            raise RuntimeError(result.error_message)
        return json.loads(result.extracted_content)
