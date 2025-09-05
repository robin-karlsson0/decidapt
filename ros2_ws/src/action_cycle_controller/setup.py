import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'action_cycle_controller'

setup(
    name=package_name,
    version='0.0.3',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        # Configuration files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=[
        'setuptools',
        'exodapt_robot_interfaces',
        'actions',
    ],
    zip_safe=True,
    maintainer='robin',
    maintainer_email='robin.karlsson0@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_cycle_controller = ' + package_name +
            '.action_cycle_controller:main',
            'action_decision = ' + package_name + '.action_decision:main',
        ],
    },
)
