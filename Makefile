# Makefile

MINIO_EXECUTABLE := D:\codeSoftware\minio\minio.exe
MINIO_DATA_DIR := D:\codeSoftware\minio\minioData
REDIS_DIR := D:\codeSoftware\Redis-x64-5.0.14.1

START_LLM := xinference-local

.PHONY: minio, llm, redis, redis-cli, celery, run

minio:
	$(MINIO_EXECUTABLE) server --address :9005 $(MINIO_DATA_DIR)


llm:
	@echo "Starting xinference-local..."
	@$(START_LLM)


redis:
	$(REDIS_DIR)\redis-server.exe $(REDIS_DIR)\redis.windows.conf


redis-cli:
	$(REDIS_DIR)\redis-cli.exe -h 127.0.0.1 -p 6379


celery:
	celery -A utils.celery worker --loglevel=info -P eventlet


run:
	python app.py