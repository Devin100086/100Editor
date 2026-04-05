from setuptools import find_namespace_packages, setup


setup(
    name="leftrefill",
    version="0.1.0",
    description="LeftRefill reference-guided inpainting package",
    packages=find_namespace_packages(
        where=".",
        include=["ldm*", "dataloaders*", "inpainting_ldm*", "leftrefill*"],
    ),
    py_modules=["run", "test_inpainting"],
    include_package_data=True,
    python_requires=">=3.8",
)
