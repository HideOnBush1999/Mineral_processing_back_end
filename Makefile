# Makefile

# MinIO executable and data directory paths
MINIO_EXECUTABLE := D:\codeSoftware\minio\minio.exe
MINIO_DATA_DIR := D:\codeSoftware\minio\minioData

START_LLM := xinference-local

.PHONY: minio, llm, run

minio:
	$(MINIO_EXECUTABLE) server --address :9005 $(MINIO_DATA_DIR)


# 定义启动命令的规则
llm:
	@echo "Starting xinference-local..."
	@$(START_LLM)

run:
	python app.py