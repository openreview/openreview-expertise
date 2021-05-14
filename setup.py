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
    author='Michael Spector, Carlos Mondragon',
    author_email='spector@cs.umass.edu, carlos@openreview.net',
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
        'spacy==2.1.0',
        'python-Levenshtein',
        'allennlp==0.9.0',
        'sacremoses',
        'rank_bm25',
        'pytest',
        'overrides==2.8.0'
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    zip_safe=False
)
