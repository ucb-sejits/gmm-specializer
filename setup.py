from distutils.core import setup

setup(
    name='gmm',
    version='0.95a',

    packages=[
        'gmm',
    ],

    package_data={
        'gmm': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
    ]
)

