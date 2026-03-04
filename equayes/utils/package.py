import os
import shutil
import subprocess
import sys


def main():
    required_packages = ["sys", "os", "subprocess", "setuptools", "build", "twine"]
    missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]

    if missing_packages:
        print(f"Fehlende Pakete: {', '.join(missing_packages)}")
        print("Bitte installieren Sie die fehlenden Pakete mit:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

    try:
        shutil.rmtree("./dist", ignore_errors=True)
        subprocess.run([sys.executable, "-m", "build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Bauen des Pakets: {e}")
        sys.exit(1)

    print("Paket erfolgreich erstellt.")


def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
