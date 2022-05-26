OUTPUT_DIST="output_dist"
APP_NAMES="corva-welleng"
IGNORED_FILES="requirements*.txt setup.cfg .gitignore Makefile resources/test_data test README.md"
SOURCE="welleng_updated"

## help: Show this help.
.PHONY: helpccscscs
help: Makefile
	@sed -n 's/^##\s//p' $<

## install: Install all requirements.
.PHONY: install
install: install-requirements

## install-requirements: Install requirements.
.PHONY: install-requirements
install-requirements:
	@pip install -U pip
	@pip install .

## test: Run tests and show code coverage.
.PHONY: test
test:
	@coverage run --source=${SOURCE} -m unittest -v

## coverage-html: Show code coverage HTML report.
.PHONY: coverage-html
coverage-html:
	@coverage html
	@x-www-browser htmlcov/index.html

## coverage-report: Show code coverage report.
.PHONY: coverage-report
coverage-report:
	@coverage --source=${SOURCE} report

## lint: Run linter.
.PHONY: lint
lint:
	@flake8

## clean: Clean autogenerated files.
.PHONY: clean
clean:
	@-python3 -Bc "for p in __import__('pathlib').Path('.').rglob('*.py[co]'): p.unlink()"
	@-python3 -Bc "for p in __import__('pathlib').Path('.').rglob('__pycache__'): p.rmdir()"
	@-rm -rf welleng/*.egg-info
	@-rm -rf htmlcov
	@-rm -rf .coverage
	@-rm -rf ${OUTPUT_DIST}

## package: create a deployment package
.PHONY: package
package: clean
	@./.github/scripts/lambda_packager.sh -o ${OUTPUT_DIST} -l "no" -i "yes" -x ${IGNORED_FILES} -m .

## deploy: deploy to environment: [qa, staging, production]
.PHONY: deploy
deploy:
	@if test -z ${environment} ; then \
	 echo Provide valid environment [qa, staging, production]; \
	 exit 30; \
	fi

	@if [ ${environment} != "qa" ] && [ ${environment} != "staging" ] && [ ${environment} != "production" ]; then \
	 echo ${environment} is not a valid environment [qa, staging, production]; \
	 exit 31; \
	fi
	@./.github/scripts/lambda_deploy.sh -e ${environment} -i "${OUTPUT_DIST}" -f ${APP_NAMES}


## clean-deploy: clean, package, and deploy to environment: [qa, staging, production]
.PHONY: clean-deploy
clean-deploy: clean package deploy