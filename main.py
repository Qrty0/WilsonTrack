"""Point d'entrée de WilsonTrack.

    python main.py                → interface graphique
    python main.py analyser ...   → ligne de commande (voir « python main.py --aide »)
"""
import multiprocessing
import sys

CLI_COMMANDS = {"analyser", "comparer", "synthetique", "valider", "cameras", "nuit", "lot", "surveiller",
                "-h", "--help", "--aide"}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--autotest":
        from wilsontrack.gui.autotest import run_autotest
        return run_autotest()
    if len(sys.argv) > 1 and sys.argv[1] in CLI_COMMANDS:
        from wilsontrack.cli import main as cli_main
        args = sys.argv[1:]
        if args[0] == "--aide":
            args = ["--help"]
        return cli_main(args)
    from wilsontrack.gui.app import run_gui
    return run_gui()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
