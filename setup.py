from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mpcc_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), 
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), 
            glob('config/*.csv')),
        (os.path.join('share', package_name, 'maps'), 
            glob('maps/*.png')),
        (os.path.join('share', package_name, 'maps'), 
            glob('maps/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bhavik Jain',
    maintainer_email='bhavik.bj.1205@gmail.com',
    description='Model Predictive Contouring Control (MPCC) for F1TENTH autonomous racing',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mpcc_node = mpcc_controller.mpcc_node:main',
            
            # Utility scripts
            'extract_waypoints = mpcc_controller.utils.extract_waypoints:main',
            'visualize_track = mpcc_controller.utils.visualize_track:main',
            'tune_params = mpcc_controller.utils.tune_params:main',
        ],
    },
)
