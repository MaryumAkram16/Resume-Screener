.PHONY: verify smoke docker-build

verify:
	python scripts/verify_artifacts.py

smoke: verify
	python -m py_compile app.py scripts/verify_artifacts.py

docker-build:
	docker build -t resume-screener:local .
