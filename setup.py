from setuptools import setup

from hellflame.hellflame import __version__

setup(
    name="hellflame",
    version=__version__,
    install_requires=['pyyaml'],
    packages=['hellflame','hellflame.services','hellflame.snippets'],
    author="yuuho",
    author_email="y.horiuchi@suou.waseda.jp",
    url="https://github.com/yuuho/Hellflame",
    description="Object-Oriented research tool for deep learning",
    entry_points={
        "console_scripts": [
            "hellflame = hellflame.hellflame:main"
        ]
    }
)