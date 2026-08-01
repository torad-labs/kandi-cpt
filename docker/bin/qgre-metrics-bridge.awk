# qgre-metrics-bridge.awk — engine train.log  ->  <runDir>/metrics.jsonl
#
# WHY THIS EXISTS. The conductor telemetry reader tails <runDir>/metrics.jsonl
# (CloudRemoteTelemetry.kt:135-139, MetricsSeries.kt:23) — and the engine writes NO SUCH FILE.
# Measured on engine 9cd16822: `grep -rl "metrics.jsonl|cpt_stages.json"` over the whole tree,
# .venv excluded, returns ZERO files. The engine's only metric surface is a human log line emitted
# by train_cpt.py:1527 through log() at :94-96:
#
#   [HH:MM:SS] step 12/100 loss 3.1416 lr 1.00e-04 peakmem 11.2GB (resident ...) 1.83s/step
#    $1         $2   $3     $4   $5    $6 $7
#
# So the image must bridge, or the loss chart on the operator's screen stays permanently empty.
#
# WHAT THE READER ACCEPTS. MetricsSeries.parseDirectMetricLine (MetricsSeries.kt:122-133) needs an
# integer "step" key and takes every other NON-STRING primitive as a named series. Emitting loss
# and lr as bare numbers is therefore exactly what LossChartInput(metric="loss") reads back.
#
# The scan is positional rather than a regex capture so it runs identically under mawk and gawk,
# and it is anchored on the pair (step N/T, loss X) so ordinary prose mentioning the word "step"
# cannot manufacture a data point.

{
    for (i = 1; i <= NF - 3; i++) {
        if ($i == "step" && $(i + 2) == "loss" &&
            $(i + 1) ~ /^[0-9]+\/[0-9]+$/ && $(i + 3) ~ /^[0-9]+(\.[0-9]+)?$/) {
            split($(i + 1), a, "/")
            line = "{\"step\":" a[1] ",\"loss\":" $(i + 3)
            if ($(i + 4) == "lr" && $(i + 5) ~ /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/) {
                line = line ",\"lr\":" $(i + 5)
            }
            line = line "}"
            print line >> M
            fflush(M)
            break
        }
    }
}
