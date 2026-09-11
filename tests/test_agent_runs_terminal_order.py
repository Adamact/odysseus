import asyncio

from src import agent_runs


def test_done_is_published_only_after_generator_cleanup():
    async def scenario():
        session_id = "terminal-order-test"
        cleanup_finished = asyncio.Event()

        async def stream():
            yield 'data: {"delta":"answer"}\n\n'
            yield "data: [DONE]\n\n"
            await asyncio.sleep(0)
            cleanup_finished.set()

        run = agent_runs.start(session_id, stream())
        received = []
        async for event in agent_runs.subscribe(session_id, run):
            received.append(event)
            if event.strip() == "data: [DONE]":
                assert cleanup_finished.is_set()

        assert [event.strip() for event in received] == [
            'data: {"delta":"answer"}',
            "data: [DONE]",
        ]
        agent_runs._RUNS.pop(session_id, None)

    asyncio.run(scenario())
