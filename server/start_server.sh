#!/bin/bash

# OpenMP 라이브러리 중복 초기화 경고 해결
export KMP_DUPLICATE_LIB_OK=TRUE

# 서버 시작
python3 main.py
