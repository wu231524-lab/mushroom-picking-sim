import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'mushroom_panda_gazebo'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            os.path.join('share', package_name),
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.xacro') + glob('urdf/*.urdf'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='jay@example.com',
    description='Gazebo Classic integration for the mushroom picking robot.',
    license='AGPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            (
                'mushroom_follower = '
                'mushroom_panda_gazebo.mushroom_follower:main'
            ),
            (
                'mushroom_target_selector = '
                'mushroom_panda_gazebo.'
                'mushroom_target_selector:main'
            ),
            (
                'pick_controller = '
                'mushroom_panda_gazebo.'
                'day15_autonomous_pick:main'
            ),
        ],
    },
)
