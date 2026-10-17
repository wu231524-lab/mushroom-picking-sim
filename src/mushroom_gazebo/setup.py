from glob import glob
from setuptools import find_packages, setup

package_name = 'mushroom_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['launch/mushroom_field.launch.py']),
    ('share/' + package_name + '/worlds', ['worlds/mushroom_field.world']),
    ('share/' + package_name + '/models/wrist_camera', glob('models/wrist_camera/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='jay@todo.todo',
    description='Gazebo Classic world and camera assets for mushroom picking.',
    license='AGPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
