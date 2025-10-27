import os

import share
from dotenv import load_dotenv

load_dotenv()

LOCAL_DIR = os.getenv("LOCAL_DIR")
REMOTE_DIR = os.getenv("REMOTE_DIR")
WORK_DIR = os.getenv("WORK_DIR")
JOB_USER = os.getenv("JOB_USER")
INIT_NUM = os.getenv("INIT_NUM")
BO_NUM = os.getenv("BO_NUM")
atom_location = os.gentenv("atom_location")

share.make_instance(LOCAL_DIR, REMOTE_DIR, WORK_DIR, JOB_USER, INIT_NUM, BO_NUM)
