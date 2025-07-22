@echo off
setlocal enabledelayedexpansion

:: Définition des listes
set thresholds=0.0000002
set walkings=1 5
set multipliers=1 2

:: Boucles imbriquées
for %%t in (%thresholds%) do (
    for %%w in (%walkings%) do (
        for %%m in (%multipliers%) do (
            echo === Running: threshold=%%t walking=%%w multiplier=%%m ===

            python optimized_vrp_pipeline.py ^
                solver.threshold=%%t ^
                solver.walking=%%w ^
                solver.multiplier=%%m

            echo === Completed: threshold=%%t walking=%%w multiplier=%%m ===
        )
    )
)

endlocal