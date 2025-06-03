from setuptools import find_packages, setup

package_name = 'state_manager'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'exodapt_robot_interfaces',
        'exodapt-robot-pt',
        'transformers>=4.52.3',
    ],
    zip_safe=True,
    maintainer='robin',
    maintainer_email='robin.karlsson0@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_manager = ' + package_name + '.state_manager:main',
        ],
    },
)
