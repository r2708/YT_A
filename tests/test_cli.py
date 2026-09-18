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
