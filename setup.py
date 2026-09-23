"""Install the standalone Q1 Isaac Lab extension."""

from setuptools import find_packages, setup

setup(
    name="q1-robot-lab",
    version="0.1.0",
    description="Standalone Q1 velocity-tracking training task for Isaac Lab",
    packages=find_packages(include=["robot_lab", "robot_lab.*", "rsl_rl", "rsl_rl.*"]),
    package_data={
        "robot_lab": [
            "assets/data/q1/urdf/*.urdf",
            "assets/data/q1/meshes/*.STL",
            "assets/data/buaa-q1-v1/q1/urdf/*.urdf",
            "assets/data/buaa-q1-v1/q1/meshes/*.STL",
            "assets/data/buaa-q1-v1/q1/mjcf/*.xml",
        ]
    },
    include_package_data=True,
    python_requires=">=3.10",
    # RSL-RL is vendored in this repository because AMP is a local extension
    # of the upstream 2.3.1 API.
    install_requires=["numpy", "psutil", "toml"],
)
