# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skillspector import graph
from skillspector.cleanup import TempDirTracker, cleanup_result


async def run_graph_scan(
    target: str, *, use_llm: bool = False, output_format: str = "json"
) -> dict:
    """Invoke the public graph and release resources owned by the test caller."""
    tracker = TempDirTracker()
    with tracker.removing_on_error():
        result = await graph.ainvoke(
            {
                "input_path": target,
                "use_llm": use_llm,
                "llm_requested": use_llm,
                "output_format": output_format,
            },
            config={"callbacks": [tracker]},
        )
    cleanup_result(result)
    return result
