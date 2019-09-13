from setuptools import setup

from hellfire.hellfire import __version__

setup(
    name="hellfire",
    version=__version__,
    install_requires=['pyyaml'],
    packages=['hellfire','hellfire.services','hellfire.snippets'],
    author="yuuho",
    author_email="y.horiuchi@suou.waseda.jp",
    url="https://github.com/yuuho/Hellfire",
    description="Object-Oriented research tool for deep learning",
    entry_points={
        "console_scripts": [
            "hellfire = hellfire.hellfire:main"
        ]
    }
)