"""Point d'entrée de WilsonTrack.

    python main.py                → interface graphique
    python main.py analyser ...   → ligne de commande (voir « python main.py --aide »)
"""
import multiprocessing
import os
import sys

# L'IA de classement (scikit-learn) ne prédit que quelques traces à la fois : avec OpenMP en mode
# « attente active », ses fils tournent à vide et saturent tous les cœurs (≈ 500 % CPU sur Mac),
# ce qui ralentit toute l'application. Un seul fil, en attente passive, suffit largement.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

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
