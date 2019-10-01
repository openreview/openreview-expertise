from setuptools import setup

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
        'pytorch_pretrained_bert'
    ],
    zip_safe=False
)
