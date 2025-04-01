#!/bin/bash

cd $(dirname $0)

python3 setup.py sdist bdist_wheel 
