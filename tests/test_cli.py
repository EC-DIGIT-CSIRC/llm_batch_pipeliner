"""Tests for cli.py — argument parsing and subcommand dispatch."""

from llm_batch_pipeline.cli import build_parser, main


class TestBuildParser:
    def test_creates_parser(self):
        parser = build_parser()
        assert parser is not None

    def test_subcommands_registered(self):
        parser = build_parser()
        # Verify all subcommands parse without error
        for cmd in ["list"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_init_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["init", "test_run", "--plugin", "spam_detection"])
        assert args.command == "init"
        assert args.name == "test_run"
        assert args.plugin_name == "spam_detection"

    def test_run_subcommand_with_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--plugin",
                "spam_detection",
                "--model",
                "llama3",
                "--backend",
                "ollama",
                "--auto-approve",
                "--dry-run",
            ]
        )
        assert args.command == "run"
        assert args.plugin_name == "spam_detection"
        assert args.model == "llama3"
        assert args.backend == "ollama"
        assert args.auto_approve is True
        assert args.dry_run is True


class TestMain:
    def test_no_args_shows_help(self):
        rc = main([])
        assert rc == 0

    def test_version(self, capsys):
        import pytest

        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0

    def test_list_command(self):
        rc = main(["list"])
        assert rc == 0

    def test_init_creates_directory(self, tmp_path):
        rc = main(["init", "test_run", "--plugin", "spam_detection", "--batch-jobs-root", str(tmp_path)])
        assert rc == 0
        # Check directory was created
        dirs = list(tmp_path.iterdir())
        assert len(dirs) == 1
        assert "test_run" in dirs[0].name
        assert (dirs[0] / "config.toml").is_file()
        assert (dirs[0] / "input").is_dir()

    def test_init_auto_numbers_sequential(self, tmp_path):
        main(["init", "dup", "--plugin", "spam_detection", "--batch-jobs-root", str(tmp_path)])
        main(["init", "dup", "--plugin", "spam_detection", "--batch-jobs-root", str(tmp_path)])
        dirs = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        assert dirs == ["batch_001_dup", "batch_002_dup"]
