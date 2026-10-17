from setuptools import find_packages, setup

package_name = 'mushroom_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/best.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='jay@todo.todo',
    description='RGB-D mushroom detection and localization nodes.',
    license='AGPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mushroom_detector = mushroom_vision.mushroom_detector:main',
            'mushroom_3d_localizer = mushroom_vision.mushroom_3d_localizer:main',
            'mushroom_world_localizer = mushroom_vision.mushroom_world_localizer:main',
            'd405_tcp_alignment = mushroom_vision.d405_tcp_alignment:main',
            'd405_step_proposal = mushroom_vision.d405_step_proposal:main',
            'd405_step_executor = mushroom_vision.d405_step_executor:main',
        ],
    },
)
