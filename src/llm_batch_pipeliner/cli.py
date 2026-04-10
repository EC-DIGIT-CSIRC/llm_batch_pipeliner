# ... existing content before CLI - keep it

# -- subcommand handlers continuing -----------------------------------------------------------------
def _cmd_test_inference(args: argparse.Namespace) -> int:
    """Handle ``test-inference`` subcommand."""
    from llm_batch_pipeline.stages import (
        stage_discover,
        stage_filter_1,
        stage_transform,
        stage_validate,
    )
    
    config = _build_config(args)