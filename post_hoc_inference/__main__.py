"""
THIS FILE IS A SHOWCASE FOR A NICE CODEBASE ENTRYPOINT WITH CLI ARGS!
This python skript is meant to be the potential main entry point, in case this
package is self-consistent and/or should be executable from a central point.
"""

from post_hoc_inference.utils import log

__author__ = "Christian Findenig"
__email__ = "christian.findenig@mcl.at"
__copyright__ = "Copyright 2026, Materials Center Leoben Forschung GmbH"
__license__ = "MIT"
__status__ = "Development"


logger = log.getLogger(__name__)


def main(*args):
    """
    The main starting point when running the post_hoc_inference package.

    Since this is a library, nothing happens
    """
    print(
        "Post-Hoc Inference entry point. This project is a library. More information are provided in the Readme.md or 'post_hoc_inference.core.post_hoc_inference.PostHocInference'"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PostHocInference tool")
    parser.add_argument("-t", "--target", type=int, default=20, help="Sample argument for demonstration.")

    args = parser.parse_args()
    main(args.target)
