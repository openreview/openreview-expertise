from setuptools import setup

setup(
    name='openreview-expertise',
    version='1.0',
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
        'numpy==1.26.4',
        'scipy',
        'adapter-transformers==3.2.1post0',
        'pandas',
        'nltk',
        'gensim>=4.2.0',
        'torch',
        'cloudpickle',
        'scikit-learn',
        'tqdm',
        'pytorch_pretrained_bert',
        'ipdb',
        'spacy==3.7.2',
        'en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz ',
        'python-Levenshtein',
        'sacremoses',
        'rank_bm25',
        'pytest',
        'overrides==3.1.0',
        'flask==2.2.2',
        'flask-cors==3.0.9',
        'cffi>=1.0.0',
        'celery==5.2.7',
        "kombu>=5.3.0,<6.0",
        'redis',
        'pytest-celery',
        'shortuuid',
        'redisai',
        'python-dotenv',
        'importlib-metadata==4.13.0',
        'werkzeug==2.2.2'
    ],
    zip_safe=False
)