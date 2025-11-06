from setuptools import find_packages, setup


setup(
	name="kernel-course",
	version="0.1.0",
	description="Educational kernels implemented in Python and other backends.",
	long_description_content_type="text/markdown",
	author="kernel-course contributors",
	license="MIT",
	packages=find_packages(include=["kernel_course", "kernel_course.*"]),
	include_package_data=True,
	python_requires=">=3.8",
	install_requires=[],
)
