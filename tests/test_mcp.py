"""The MCP server, through an in-process client - the same path a real client takes."""
import asyncio
import json

import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from drugagent import tools  # noqa: E402
from drugagent.mcp_server import server  # noqa: E402


def call(name, args):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, args)
    return asyncio.run(go())


def payload(result):
    return json.loads(result.content[0].text)


def test_every_registry_tool_is_served_plus_verify_text():
    async def go():
        async with Client(server) as client:
            return {t.name for t in (await client.list_tools()).tools}
    assert asyncio.run(go()) == set(tools.REGISTRY) | {"verify_text"}


def test_descriptors_come_back_keyed_by_method():
    result = payload(call("compute_descriptors", {"compound": "aspirin"}))
    assert result["hba"] == {"lipinski": 4, "rdkit": 3, "pubchem": 4}
    assert result["logp"] == {"crippen": 1.31, "xlogp3": 1.2}


def test_verify_text_flags_a_wrong_number():
    result = payload(call("verify_text", {"text": "Ibuprofen has a TPSA of 83.6 and a molecular weight of 206.28."}))
    assert result["summary"]["contradicted"] == 1 and result["summary"]["supported"] == 1


def test_bad_input_reaches_the_client_as_a_readable_error():
    result = call("compute_descriptors", {"compound": "unobtainium"})
    assert result.is_error and "cannot resolve 'unobtainium'" in result.content[0].text
