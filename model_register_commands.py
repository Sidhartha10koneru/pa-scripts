python freeze_test_set.py --run exp_nofilter_2022 --name test_v1_2026-05_2026-07 --out frozen/test_v1_2026-05_2026-07

R="--frozen frozen/test_v1_2026-05_2026-07"
python register_run.py exp_nofilter_2022 $R --status benchmark --qc qckeep  --seed 42 --notes "base, no filter, 2022-26"
python register_run.py exp_qc_exclude_rv $R --status candidate --qc qcexcl  --seed 42 --notes "quality codes excluded"
python register_run.py exp_qc_false_rv   $R --status archived  --qc qcfalse --seed 42 --notes "quality FALSE (v6); worse than keep"
python register_run.py exp_keep_s7       $R --status archived  --qc qckeep  --seed 7  --notes "seed check"
python register_run.py exp_exclude_s7    $R --status archived  --qc qcexcl  --seed 7  --notes "seed check"
