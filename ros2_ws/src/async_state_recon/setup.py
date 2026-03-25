from setuptools import find_packages, setup

package_name = 'async_state_recon'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'exodapt_robot_interfaces'],
    zip_safe=True,
    maintainer='Robin Karlsson',
    maintainer_email='robin.karlsson0@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'asr_manager = ' + package_name + '.asr_manager:main',
        ],
    },
)
