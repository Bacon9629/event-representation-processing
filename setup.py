from setuptools import setup, find_packages
import os

# 讀取 README.md 作為長描述
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='event-representation-processing',
    version='0.1.0',
    author='Bacon9629',
    author_email='qwe43213652@example.com',  # 請修改為實際聯絡信箱
    description='A package for processing event-based representations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Bacon9629/event-representation-processing',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',  # 或其他適合的狀態
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # 根據實際情況修改授權條款
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "dv",
        "opencv-python",
        "matplotlib",
        "dv_processing",
        "torch",
    ],
)
