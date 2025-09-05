@echo OFF
REM This script runs all Python analyses for the assignment and logs their output.

ECHO Starting all analyses...

ECHO Creating the 'logs' directory if it doesn't exist...
if not exist logs mkdir logs

ECHO.
ECHO [1/5] Running Q1 Part 1: Stationary Environment Analysis...
pushd Q1\part1
python 1_2.py > ..\..\logs\log_Q1_part_1.txt 2>&1
popd
ECHO Done.

ECHO.
ECHO [2/5] Running Q1 Part 2: Non-Stationary Environment Analysis...
pushd Q1\part2
python 1_3.py > ..\..\logs\log_Q1_part_2.txt 2>&1
popd
ECHO Done.

ECHO.
ECHO [3/5] Running Q1 Part 3: Prioritized VI Analysis...
pushd Q1\part3
python 1_4.py > ..\..\logs\log_Q1_part_3.txt 2>&1
popd
ECHO Done.

ECHO.
ECHO [4/5] Running Q2 Part 1: Online Knapsack Problem...
pushd Q2\part1
python 2.py > ..\..\logs\log_Q2_part_1.txt 2>&1
popd
ECHO Done.

ECHO.
ECHO [5/5] Running Q2 Part 2: Portfolio Optimization...
pushd Q2\part2
python 3.py > ..\..\logs\log_Q2_part_2.txt 2>&1
popd
ECHO Done.

ECHO.
ECHO =================================
ECHO All scripts finished.
ECHO Logs are available in the '..\logs' directory relative to each part's folder.
ECHO =================================
