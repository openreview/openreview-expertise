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
        'en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz ',
        'python-Levenshtein',
        'allennlp==0.9.0',
        'sacremoses',
        'rank_bm25',
        'pytest',
        'overrides==2.8.0',
        'flask',
        'flask-cors==3.0.9',
        'cffi>=1.0.0',
        'celery',
        'redis',
        'pytest-celery',
        'shortuuid',
        'redisai',
        'python-dotenv',
        'importlib-metadata==4.13.0',
        "Werkzeug==2.3.7"
    ],
    zip_safe=False
)
