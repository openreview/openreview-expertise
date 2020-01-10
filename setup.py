from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        check_call('python -m spacy download en'.split())
        check_call('python -m spacy download en_core_web_sm'.split())
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        check_call('python -m spacy download en'.split())
        check_call('python -m spacy download en_core_web_sm'.split())
        install.run(self)

setup(
    name='openreview-expertise',
    version='0.1',
    description='OpenReview paper-reviewer affinity modeling',
    url='https://github.com/iesl/openreview-evidence',
    author='Michael Spector',
    author_email='spector@cs.umass.edu',
    license='MIT',
    packages=[
        'expertise'
    ],
    install_requires=[
        'openreview-py>=1.0.1',
        'numpy',
        'pandas',
        'nltk',
        'gensim',
        'torch',
        'cloudpickle',
        'scikit-learn',
        'tqdm',
        'pytorch_pretrained_bert',
        'ipdb',
        'spacy',
        'python-Levenshtein'
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    zip_safe=False
)
