# The loom website lives under docs/ ( VitePress ), with its own package.json, so the project root
# stays a Python package root. `npm install` is idempotent, so there is no separate step to remember.

.PHONY: help site site-build anim

help:
	@echo "  make site        the website, live reloading   ( docs/, VitePress )"
	@echo "  make site-build  the website into docs/.vitepress/dist"
	@echo "  make anim        regenerate the tutorials' animations ( docs/public/anim/*.gif ) -- needs jax + matplotlib"

site: docs/node_modules
	cd docs && npm run dev

site-build: docs/node_modules
	cd docs && npm run build

docs/node_modules: docs/package.json
	cd docs && npm install
	@touch $@

# The animations are committed, so building the site does not need jax: this is for when an example
# changes. Run it in an environment that has loom, jax and matplotlib.
anim:
	python docs/scripts/splats_fit.py
	python docs/scripts/diffusion_inversion.py
