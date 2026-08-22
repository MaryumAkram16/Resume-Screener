.PHONY: verify smoke train-stage1 train-stage2 record-run docker-build

verify:
	python scripts/verify_artifacts.py

smoke: verify
	python -m py_compile app.py scripts/verify_artifacts.py scripts/train_stage1.py scripts/train_stage2.py scripts/record_run.py

train-stage1:
	python scripts/train_stage1.py --csv $(CSV) --output-dir artifacts/stage1

train-stage2:
	python scripts/train_stage2.py --csv $(CSV) --output-dir artifacts/stage2

record-run:
	python scripts/record_run.py --artifact-dir $(ARTIFACT_DIR) --metrics $(ARTIFACT_DIR)/metrics.json

docker-build:
	docker build -t resume-screener:local .
