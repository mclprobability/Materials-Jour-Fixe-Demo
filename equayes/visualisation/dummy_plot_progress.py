import matplotlib

matplotlib.use("Agg")  # no GUI-Backend

import matplotlib.pyplot as plt
from equayes.utils.git_info import get_git_info
import inspect
import os

def plot_score_progress(scores: list[int]):
    """
    Plots the score progression and saves it to a file.
    """
    if not scores:
        print("No scores to plot.")
        return

    # Git Infos
    git_info = get_git_info()

    # Source Infos
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)[1]
    filename = os.path.basename(caller_frame.filename)
    lineno = caller_frame.lineno
    funcname = caller_frame.function
    classname = caller_frame.frame.f_locals.get('self', None)
    classname = classname.__class__.__name__ if classname else "-"

    # Title (Git Info)
    title = f"Dice Game Progress ({git_info['version']} @ {git_info['branch']} - {git_info['commit']})"

    # Source Info (Filename, Class, Function, Line)
    source_info = f"Generated from: {filename} | Class: {classname} | Function: {funcname} | Line: {lineno}"

    rounds = list(range(1, len(scores) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, scores, marker='o')
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()

    # Fußnote
    plt.figtext(
        0.5, -0.05,
        f"{source_info}\nVersion: {git_info['version']} | Branch: {git_info['branch']} | Commit: {git_info['commit']}",
        wrap=True, horizontalalignment='center', fontsize=8
    )

    plt.savefig("dice_progress.png", bbox_inches="tight")
    print("Plot saved as dice_progress.png")
    print(source_info)
