# The loom website lives under docs/ ( VitePress ), with its own package.json, so the project root
# stays a Python package root. `npm install` is idempotent, so there is no separate step to remember.

.PHONY: help site site-build site-deploy anim version bump_version wheels check_wheels publish

help:
	@echo "  make site        the website, live reloading   ( docs/, VitePress )"
	@echo "  make site-build  the website into docs/.vitepress/dist"
	@echo "  make site-deploy site-build, then push it to the gh-pages branch ( https://hleclerc.github.io/loom/ )"
	@echo "  make version     print the version"
	@echo "  make bump_version write today's next version ( TO=x.y.z.n to choose it )"
	@echo "  make wheels      the loom-ffi wheel into dist/ ( pure Python: one for every platform ), checked"
	@echo "  make check_wheels install this version's wheel in a clean venv and check it"
	@echo "  make publish     upload this version's wheel of dist/ ( TWINE_PASSWORD = your PyPI token )"
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


# ── releasing ──
# reminders: the work is in loom ( `python -m loom.release`, docs/guide/releasing.md in loom ), configured by [tool.loom.release]
PYTHON ?= python

version:
	$(PYTHON) -m loom.release show_version

bump_version:
	$(PYTHON) -m loom.release bump_version $(if $(TO),--to $(TO))

wheels:
	$(PYTHON) -m loom.release make_wheels

check_wheels:
	$(PYTHON) -m loom.release check_wheels

publish:
	$(PYTHON) -m loom.release publish_wheels
