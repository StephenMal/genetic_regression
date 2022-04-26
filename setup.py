from setuptools import setup, find_packages

setup(
    name='geneticregression',
    version='0.1.4',
    description='Genetic Regression',
    url='https://github.com/StephenMal/genetic_regression',
    author='Stephen Maldonado',
    author_email='genetic_regression@stephenmal.com',
    packages=find_packages(),
    install_requires=[\
        'simpgenalg==0.2.4'\
        ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Utilities'
    ]
)
