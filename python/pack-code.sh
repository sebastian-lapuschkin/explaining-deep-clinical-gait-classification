#!/bin/bash
if [ -f code.zip ];
then
    rm code.zip
fi
zip code.zip eval_score_logs.py gait_experiments.py training.py
