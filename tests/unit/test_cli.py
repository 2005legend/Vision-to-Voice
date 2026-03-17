"""Unit tests for board_reader.cli (task 13.1)."""

from unittest.mock import MagicMock, patch

import pytest

from board_reader.cli import main


def _run_main(*argv):
    """Patch sys.argv and call main()."""
    with patch("sys.argv", ["board_reader", *argv]):
        main()


# ---------------------------------------------------------------------------
# start subcommand
# ---------------------------------------------------------------------------

def test_start_subcommand_accepted():
    """start subcommand calls load_config and SessionManager.start."""
    mock_config = MagicMock()
    mock_manager = MagicMock()

    with (
        patch("board_reader.cli.load_config", return_value=mock_config) as mock_load,
        patch("board_reader.cli.SessionManager", return_value=mock_manager),
    ):
        _run_main("start")

    mock_load.assert_called_once_with("config.yaml")
    mock_manager.start.assert_called_once_with(mock_config)


def test_start_with_custom_config():
    """--config flag passes the given path to load_config."""
    mock_config = MagicMock()
    mock_manager = MagicMock()

    with (
        patch("board_reader.cli.load_config", return_value=mock_config) as mock_load,
        patch("board_reader.cli.SessionManager", return_value=mock_manager),
    ):
        _run_main("start", "--config", "custom.yaml")

    mock_load.assert_called_once_with("custom.yaml")


def test_start_calls_stop_on_keyboard_interrupt():
    """KeyboardInterrupt during start still calls manager.stop()."""
    mock_config = MagicMock()
    mock_manager = MagicMock()
    mock_manager.start.side_effect = KeyboardInterrupt

    with (
        patch("board_reader.cli.load_config", return_value=mock_config),
        patch("board_reader.cli.SessionManager", return_value=mock_manager),
    ):
        _run_main("start")

    mock_manager.stop.assert_called_once()


# ---------------------------------------------------------------------------
# stop subcommand
# ---------------------------------------------------------------------------

def test_stop_subcommand_accepted(capsys):
    """stop subcommand prints a message and exits cleanly."""
    _run_main("stop")
    captured = capsys.readouterr()
    assert "SIGTERM" in captured.out or "Ctrl+C" in captured.out


# ---------------------------------------------------------------------------
# unknown subcommand
# ---------------------------------------------------------------------------

def test_unknown_subcommand_rejected():
    """An unknown subcommand causes SystemExit."""
    with pytest.raises(SystemExit):
        _run_main("unknown")


def test_no_subcommand_rejected():
    """Calling with no subcommand causes SystemExit."""
    with pytest.raises(SystemExit):
        _run_main()
