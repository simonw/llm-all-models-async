import asyncio
import llm
import llm_make_all_models_async  # noqa: F401 - triggers monkey-patching
from llm_make_all_models_async import AsyncModelWrapper


def test_mrchatterbox_has_async_wrapper():
    """mrchatterbox (sync-only) should get an async wrapper."""
    async_model = llm.get_async_model("mrchatterbox")
    assert async_model is not None
    assert async_model.model_id == "mrchatterbox"
    assert isinstance(async_model, AsyncModelWrapper)


async def test_mrchatterbox_async_produces_output():
    """The async mrchatterbox wrapper should produce text."""
    async_model = llm.get_async_model("mrchatterbox")
    response = async_model.prompt("What is the capital of France?")
    text = await response.text()
    assert len(text) > 0
    print(f"mrchatterbox response: {text!r}")


async def test_mrchatterbox_async_streaming():
    """The async mrchatterbox wrapper should stream chunks."""
    async_model = llm.get_async_model("mrchatterbox")
    response = async_model.prompt("Hello")
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert len(full_text) > 0
    print(f"mrchatterbox streamed {len(chunks)} chunks: {full_text!r}")
