# The loom website lives under docs/ ( VitePress ), with its own package.json, so the project root
# stays a Python package root. `npm install` is idempotent, so there is no separate step to remember.

.PHONY: help site site-build site-deploy anim wheels publish

help:
	@echo "  make site        the website, live reloading   ( docs/, VitePress )"
	@echo "  make site-build  the website into docs/.vitepress/dist"
	@echo "  make site-deploy site-build, then push it to the gh-pages branch ( https://hleclerc.github.io/loom/ )"
	@echo "  make wheels      the loom-ffi wheel ( pure Python: one for every platform ) into dist/"
	@echo "  make publish     upload dist/ to PyPI ( TWINE_PASSWORD = your PyPI token )"
	@echo "  make anim        regenerate the tutorials' animations ( docs/public/anim/*.gif ) -- needs jax + matplotlib"

site: docs/node_modules
	cd docs && npm run dev

site-build: docs/node_modules
	cd docs && npm run build

# The built site becomes the only commit of the gh-pages branch ( force-pushed: no history to keep ).
# .nojekyll, or GitHub Pages drops the _-prefixed files of VitePress. Once, on GitHub:
# Settings > Pages > Source = "Deploy from a branch", branch gh-pages, folder / ( root ).
SITE_REMOTE ?= $(shell git remote get-url origin)
SITE_BRANCH ?= gh-pages

site-deploy: site-build
	cd docs/.vitepress/dist && touch .nojekyll && rm -rf .git \
	  && git init -q -b $(SITE_BRANCH) \
	  && git add -A \
	  && git commit -q -m "site: $$(cd ../../.. && git rev-parse --short HEAD)" \
	  && git push -f $(SITE_REMOTE) $(SITE_BRANCH) \
	  && rm -rf .git

docs/node_modules: docs/package.json
	cd docs && npm install
	@touch $@

# The animations are committed, so building the site does not need jax: this is for when an example
# changes. Run it in an environment that has loom, jax and matplotlib.
anim:
	python docs/scripts/splats_fit.py
	python docs/scripts/diffusion_inversion.py


# ── the wheel ──
# loom-ffi is pure Python, so it is built anywhere. The sdot wheels require at least this version: they are released together,
# by sdot-ffi ( `scripts/wheels.py --version next` there writes both ), which is where the version is bumped.

PYTHON ?= python

wheels:
	rm -f dist/*.whl
	$(PYTHON) -m pip wheel --no-deps -w dist .

publish:
	$(PYTHON) -m twine upload --skip-existing --repository-url https://upload.pypi.org/legacy/ dist/*.whl
