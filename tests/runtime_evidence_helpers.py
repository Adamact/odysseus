"""Dispatcher doubles must simulate the receipt boundary as well as the result."""
from src.agent_runtime.journal import mark_dispatch, record_action


def authoritative_executor(function):
    @record_action
    async def execute(block, *args, **kwargs):
        mark_dispatch()
        return await function(block, *args, **kwargs)
    return execute
