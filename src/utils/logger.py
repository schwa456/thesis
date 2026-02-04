import logging
import sys
import os
from datetime import datetime, timedelta, timezone

def setup_logger(mode, exp_name, log_dir="logs"):
    # 1. 로그 저장할 폴더 생성
    os.makedirs(log_dir, exist_ok=True)

    # 2. 로거 생성
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        return root_logger

    root_logger.setLevel(logging.DEBUG)

    if root_logger.handlers:
        return root_logger
    
    def kst_converter(*args):
        utc_dt = datetime.now(timezone.utc)
        kst_dt = utc_dt + timedelta(hours=9)
        return kst_dt.timetuple()

    # 3. Format 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    formatter.converter = kst_converter

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(f"{log_dir}/exp_logs/{today}", exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/exp_logs/{today}/{mode}_log_{exp_name}.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return root_logger
