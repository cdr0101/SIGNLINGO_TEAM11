import setuptools

setuptools.setup(
    name='SIGNLINGO',
    version='0.1.0',
    description='Python project',
    author='Chelsi',
    author_email='cchelsi_be21@thapar.edu',
    url='',
    packages=setuptools.find_packages(),
    setup_requires=['nltk', 'joblib','click','regex','sqlparse','setuptools'],
)