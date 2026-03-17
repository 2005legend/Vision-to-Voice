"""CLI entry point for IntelliAgent Board Reader."""

import argparse

from board_reader.config import load_config
from board_reader.session import SessionManager


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="board_reader",
        description="IntelliAgent Board Reader — AI-powered assistive board reader.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # explain subcommand (single image → voice)
    explain_parser = subparsers.add_parser(
        "explain", help="Explain a board image and speak the result."
    )
    explain_parser.add_argument("image", help="Path to the board image (jpg, png, etc.)")
    explain_parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="CONFIG",
        help="Path to config.yaml (default: config.yaml)",
    )
    explain_parser.add_argument(
        "--mode",
        choices=["board", "diagram"],
        default="board",
        help="'board' (default): OCR+NIM+Gemini. 'diagram': Gemini vision only.",
    )

    # start subcommand (live camera session)
    start_parser = subparsers.add_parser("start", help="Start a live board-reading session.")
    start_parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="CONFIG",
        help="Path to config.yaml (default: config.yaml)",
    )

    # stop subcommand
    subparsers.add_parser("stop", help="Stop a running board-reading session.")

    args = parser.parse_args()

    if args.command == "explain":
        from board_reader.explain import explain_image
        config = load_config(args.config)
        explain_image(args.image, config, mode=args.mode)

    elif args.command == "start":
        config = load_config(args.config)
        manager = SessionManager()
        try:
            manager.start(config)
        except KeyboardInterrupt:
            pass
        finally:
            manager.stop()

    elif args.command == "stop":
        print("Send SIGTERM to the running process or use Ctrl+C to stop the session.")

    else:
        parser.print_help()
        raise SystemExit(1)
