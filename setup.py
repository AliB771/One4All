from setuptools import setup, find_packages

setup(
    name="ml-router-expert",
    version="0.1.0",
    description="ML Router-Expert System with LoRA",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0", 
        "peft>=0.4.0",
        "pyyaml>=6.0",
    ],
)
