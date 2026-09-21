from pathlib import Path

from typer.testing import CliRunner

from video_dataset import cli

runner = CliRunner()


def _reset_state() -> None:
    cli.state.config = None
    cli.state.config_path = None
    cli.state.overrides = {}
    cli.state.log_level = None


def test_stages_command():
    _reset_state()
    result = runner.invoke(cli.app, ["stages"])
    assert result.exit_code == 0
    assert "DOWNLOAD" in result.output and "EXPORT" in result.output


def test_show_config_with_set_after_command(tmp_path: Path):
    _reset_state()
    result = runner.invoke(cli.app, ["show-config", "--set", "vision.provider=mock", "--set", f"project.data_dir={tmp_path}"])
    assert result.exit_code == 0, result.output
    assert "provider: mock" in result.output


def test_status_on_empty_db(tmp_path: Path):
    _reset_state()
    result = runner.invoke(cli.app, ["--data-dir", str(tmp_path / "d"), "status"])
    assert result.exit_code == 0, result.output
    assert "No videos registered" in result.output


def test_run_rejects_missing_input(tmp_path: Path):
    _reset_state()
    result = runner.invoke(cli.app, ["run", str(tmp_path / "nope.txt"), "--set", f"project.data_dir={tmp_path / 'd'}"])
    assert result.exit_code != 0


def test_load_env_file_skips_blank_values(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_HOME=\nHF_TOKEN=\nMY_KEY=abc\nEXISTING=from-file\n")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "from-env")
    applied = cli.load_env_file(env)
    assert applied == ["MY_KEY"]
    assert "HF_HOME" not in __import__("os").environ  # blank line must not export an empty HF_HOME
    assert __import__("os").environ["MY_KEY"] == "abc" and __import__("os").environ["EXISTING"] == "from-env"
    assert cli.load_env_file(tmp_path / "missing.env") == []
