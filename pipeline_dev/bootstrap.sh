#! /bin/bash

# first switch to directory of this script (in case it is started via double click from Explorer)
cd "$(dirname "$0")"

# set pipefail
set -e

# create venv, works for win and linux
venv_name=".venv"
python3 -m venv $venv_name 2> /dev/null || python -m venv $venv_name
[ -f $venv_name/bin/activate ] && . $venv_name/bin/activate
[ -f $venv_name/Scripts/activate ] && . $venv_name/Scripts/activate


pip install -r requirements.txt
