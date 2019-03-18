from setuptools import setup

setup(
    name='openreview-expertise',
    version='0.0',
    description='OpenReview paper-reviewer affinity modeling',
    url='https://github.com/iesl/openreview-evidence',
    author='Michael Spector',
    author_email='spector@cs.umass.edu',
    license='MIT',
    packages=[
        'expertise'
    ],
    install_requires=[
        'openreview-py',
        'numpy',
        'pandas',
        'nltk',
        'gensim',
        'torch',
        'cloudpickle',
        'scikit-learn',
        'tqdm'
    ],
    zip_safe=False
)
