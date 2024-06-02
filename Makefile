# Makefile

# MinIO executable and data directory paths
MINIO_EXECUTABLE := D:\codeSoftware\minio\minio.exe
MINIO_DATA_DIR := D:\codeSoftware\minio\minioData

.PHONY: start_minio

# Target to start MinIO server
start_minio:
	$(MINIO_EXECUTABLE) server $(MINIO_DATA_DIR)
