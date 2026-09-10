# test/test_mcp_web_search.py
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from config.bailian_mcp_config import mcp_config
from tool.logger import logger


async def test_raw_http():
    """第一步：用原始 HTTP 请求测试端点是否可达"""
    logger.info("=" * 50)
    logger.info("[诊断 1] 原始 HTTP 请求测试")
    logger.info(f"  URL: {mcp_config.mcp_base_url}")
    logger.info("=" * 50)

    headers = {
        "Authorization": f"Bearer {mcp_config.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    initialize_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0",
            },
        },
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                mcp_config.mcp_base_url,
                headers=headers,
                json=initialize_payload,
            )
            logger.info(f"  HTTP Status: {resp.status_code}")
            logger.info(f"  Response Body: {resp.text[:500]}")

            if resp.status_code == 200:
                logger.info("[PASS] 端点可达")
            else:
                logger.error(f"[FAIL] 状态码: {resp.status_code}")
        except Exception as e:
            logger.error(f"[FAIL] 请求异常: {type(e).__name__}: {e}")


async def test_mcp_direct():
    """第二步：用 mcp 官方库直接调用（绕过 openai-agents）"""
    logger.info("")
    logger.info("=" * 50)
    logger.info("[诊断 2] mcp 官方库直接调用测试")
    logger.info("=" * 50)

    try:
        from mcp.client.streamable_http import streamable_http_client
        from mcp import ClientSession
    except ImportError as e:
        logger.error(f"[FAIL] mcp 库导入失败: {e}")
        return

    headers = {"Authorization": f"Bearer {mcp_config.api_key}"}
    http_client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(15.0))

    try:
        async with streamable_http_client(
            mcp_config.mcp_base_url,
            http_client=http_client,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init_result = await session.initialize()
                logger.info(f"[PASS] initialize 成功: {init_result}")

                tools = await session.list_tools()
                logger.info(f"[PASS] 工具列表: {[t.name for t in tools.tools]}")

                test_query = "今天天气怎么样"
                logger.info(f"  测试查询: {test_query}")
                result = await session.call_tool(
                    "bailian_web_search",
                    arguments={"query": test_query, "count": 3},
                )
                if result.content:
                    raw_text = result.content[0].text
                    data = json.loads(raw_text)
                    pages = data.get("pages") or []
                    logger.info(f"[PASS] 搜索返回 {len(pages)} 条结果")
                    for i, page in enumerate(pages, 1):
                        logger.info(f"  [{i}] {page.get('title', '无标题')}")
                        logger.info(f"      URL: {page.get('url', '')}")
                        logger.info(f"      摘要: {(page.get('snippet') or '')[:80]}...")
                else:
                    logger.warning("[WARN] 返回结果为空")

    except Exception as e:
        logger.error(f"[FAIL] mcp 直接调用失败: {type(e).__name__}: {e}")
    finally:
        await http_client.aclose()


async def main():
    await test_raw_http()
    await test_mcp_direct()
    logger.info("")
    logger.info("=" * 50)
    logger.info("全部诊断测试结束")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
