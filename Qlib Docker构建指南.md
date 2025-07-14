# Qlib Docker构建指南

## 前置条件
- 已安装Docker 20.10+
- 磁盘空间 ≥5GB

## 标准构建流程
```bash:%2FUsers%2Fzhaohua%2Fstudy%2Fqlib%2Fbuild_docker_image.sh
# 完整构建命令（包含单元测试）
./build_docker_image.sh --build-arg PYTHON_VERSION=3.8 --test