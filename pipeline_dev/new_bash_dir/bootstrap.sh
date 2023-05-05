#! bin/bash

python -m venv meow_venv

. meow_venv/Scripts/activate

pip install -r requirements.txt

python meopy.py