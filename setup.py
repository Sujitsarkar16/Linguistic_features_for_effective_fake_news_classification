import setuptools

with open('README.md', 'r', encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Linguistic_features_for_effective_fake_news_classification"
AUTHOR_USER_NAME = "Sujitsarkar16"
SRC_REPO = "fnClassification"
AUTHOR_EMAIL = "sarkarsujit9052@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Fake news classification using linguistic features and machine leaarning",
    long_description=long_description,
    long_description_content="text/markdown",
    url="https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")

)
