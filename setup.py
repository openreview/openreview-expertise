from setuptools import setup

setup(
    name='openreview-expertise',
    version='2.1.2',
    description='OpenReview paper-reviewer affinity modeling',
    url='https://github.com/openreview/openreview-expertise',
    author='OpenReview',
    author_email='info@openreview.net',
    license='MIT',
    packages=[
        'expertise'
    ],
    install_requires=[
        'setuptools>=75.6.0',
        'openreview-py>=2.0.1',
        'numpy>=2.0,<2.3',
        'scipy',
        'adapter-transformers',
        'pandas',
        'nltk',
        'gensim==4.4.0',
        'torch',
        'scikit-learn',
        'tqdm',
        'spacy==3.8.13',
        'en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz',
        'rank_bm25',
        'flask==3.0.3',
        'flask-cors==6.0.0',
        'redis',
        'shortuuid',
        'redisai',
        'python-dotenv',
        'importlib-metadata==4.13.0',
        'google-cloud-storage',
        'google-cloud-aiplatform',
        'bullmq==2.11.0'
    ],
    extras_require={
        'dev': [
            'pytest==7.3.0',
        ],
    },
    zip_safe=False
)
