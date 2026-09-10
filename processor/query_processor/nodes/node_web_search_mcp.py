# processor/query_processor/nodes/node_web_search_mcp.py
import asyncio
import json

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        query = state.get("rewritten_query")
        result = asyncio.run(self._mcp_call(query))
        web_docs = []
        if result:
            json_str = result.content[0].text
            pages = json.loads(json_str).get("pages")
            for item in pages:
                snippet = item.get("snippet")
                url = item.get("url")
                title = item.get("title")

                web_docs.append({"title": title, "url": url, "snippet": snippet})

        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        if web_docs:
            return {"web_search_docs": web_docs}
        return {}


    async def _mcp_call(self, query: str):

        try:
            http_client = httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {mcp_config.api_key}"},
                timeout=10,
            )

            async with streamable_http_client(
                url=mcp_config.mcp_base_url,
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        name="bailian_web_search",
                        arguments={
                            "query": query,
                            "count": 5
                        }
                    )

                    return result

        except Exception as e:
            logger.error(f"MCP WebSearch 调用失败: {e}")
            return None



if __name__ == "__main__":

    init_state = {
        "session_id": "test_session",
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))


