nohup python -u pa_experiments.py baselines --train-years 2025 --test-years 2026 \
  --update-start 2026-02-01 --test-start 2026-05-01 --test-end 2026-08-01 \
  --sample 1.0 --models lgb_full --out exp_base_2025_2026 > run_base_2025_2026.log 2>&1 &
