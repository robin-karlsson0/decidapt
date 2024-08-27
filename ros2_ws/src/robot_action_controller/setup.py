from setuptools import find_packages, setup

package_name = 'robot_action_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'llm_action_interfaces',
        'robot_action_interfaces',
    ],
    zip_safe=True,
    maintainer='robin',
    maintainer_email='robin.karlsson0@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_controller = ' + package_name + '.action_controller:main',
            'action_decision = ' + package_name + '.action_decision:main',
        ],
    },
)
