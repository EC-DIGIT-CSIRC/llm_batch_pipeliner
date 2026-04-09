"""Smoke-test: package imports and version are correct."""

import llm_batch_pipeline


def test_version():
    assert llm_batch_pipeline.__version__ == "0.1.0"


def test_cli_main_no_args_returns_zero():
    from llm_batch_pipeline.cli import main

    # No subcommand shows help and returns 0
    assert main([]) == 0


def test_cli_list_returns_zero():
    from llm_batch_pipeline.cli import main

    assert main(["list"]) == 0
