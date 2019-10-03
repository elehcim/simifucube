all: _render.pyx
	python setup_extension.py build_ext --inplace

test:
	python -c "from _render import render_cube"