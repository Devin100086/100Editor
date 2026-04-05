from pathlib import Path

from setuptools import setup


def discover_packages():
    root = Path(__file__).parent
    packages = ["threestudio"]
    for init_file in root.rglob("__init__.py"):
        rel_dir = init_file.parent.relative_to(root)
        if rel_dir == Path("."):
            continue
        packages.append("threestudio." + ".".join(rel_dir.parts))
    return sorted(set(packages))


setup(
    name="hundrededitor-threestudio",
    version="0.1.0",
    description="Threestudio fork packaged for 100Editor",
    packages=discover_packages(),
    package_dir={"threestudio": "."},
    include_package_data=True,
    python_requires=">=3.10",
)
