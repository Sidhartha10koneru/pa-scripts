nohup python progress_report.py --frozen frozen/test_v1_2026-05_2026-07 \
  --runs bench_s42=exp_nofilter_2022 bench_s7=exp_keep_s7 \
         excl_s42=exp_qc_exclude_rv excl_s7=exp_exclude_s7 \
         pax_s42=exp_v8d_l6pax_s42 pax_s7=exp_v8d_l6pax_s7 \
  --corrected-from exp_v8d_l6pax_s42 --out progress > run_progress.log 2>&1 &
