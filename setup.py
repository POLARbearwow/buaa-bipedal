"""Install the standalone Q1 Isaac Lab extension."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="q1-robot-lab",
    version="0.1.0",
    description="Standalone Q1 velocity-tracking training task for Isaac Lab",
    packages=find_packages("source/robot_lab"),
    package_dir={"": "source/robot_lab"},
    package_data={
        "robot_lab": [
            "assets/data/q1/urdf/*.urdf",
            "assets/data/q1/meshes/*.STL",
            "assets/data/buaa-q1-v1/q1/urdf/*.urdf",
            "assets/data/buaa-q1-v1/q1/meshes/*.STL",
        ]
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["numpy", "psutil", "toml", "rsl-rl-lib>=2.3.3"],
)
