cat > exp_nofilter_2022/fn_2022_README.txt <<'EOF'
False negatives of the 2022-2026 model (p_lgb_full_2022) on the test set May-Jul 2026, all filing orders.
Service = CSN + FINAL_CPT. Service is PA if any line has y=1; service_score = highest line score.
False negative = service is PA but service_score < 0.80 (best-F1 cut-off). 18,070 services.

fn_2022_top_pa_guid.csv       one row per FN service: the highest-scoring TRUE line (y=1)  <- use this to review
fn_2022_top_guid.csv          one row per FN service: the highest-scoring line (y can be 0 or 1)
fn_2022_services.csv          one row per FN service, GROUP_GUIDS = all its lines, comma-separated
fn_2022_service_lines.parquet every line of the FN services (more than 18,070 rows; some lines have y=0)

Columns: GROUP_GUID = line id (joins to v4 std_data), y = label from GROUP_LABEL (1 TRUE, 0 FALSE),
p_lgb_full_2022 = model score of the line, service_score = score of the service.
Scores are rankings, not calibrated probabilities.
EOF
