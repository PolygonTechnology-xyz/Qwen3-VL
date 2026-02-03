#!/bin/bash

pip install --upgrade pip && pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation --no-cache-dir