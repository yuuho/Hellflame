from setuptools import setup

setup(
    name="hellfire",
    version="0.0.1",
    packages=['hellfire','hellfire.services'],
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