#!/bin/bash
# status.sh -- one-screen status for a running (or finished) ADflow stage.
#
#   ./status.sh                 newest log in logs/
#   ./status.sh logs/g20_it1_S2.log
#   watch -n 5 ./status.sh      live, refreshing
#
# The monitor columns ADflow prints are:
#   Res rho     mean-flow density residual
#   Res nuturb  SA turbulence residual -- this is the one that stalls in S2
#   totalRes    what L2Convergence is measured against, as a fraction of the
#               free-stream totalR0 printed at startup
#   Y+_max      largest first-cell y+ anywhere on the viscous wall
cd "$(dirname "$0")"
LOG=${1:-$(ls -t logs/*_S[12].log 2>/dev/null | head -1)}
[ -f "$LOG" ] || { echo "no log found"; exit 1; }

# -x mpiexec, NOT -f on the script name: a bash wrapper whose command
# line mentions adflow_run2.py matches -f and makes a wait loop hang on
# itself.  The solver is always an mpiexec process.
PID=$(pgrep -x mpiexec | head -1)
if [ -n "$PID" ]; then
    STATE="RUNNING  (pid $PID, elapsed $(ps -o etime= -p "$PID" | tr -d ' '))"
else
    STATE="not running"
fi

NCYC=$(sed -nE 's/.*Grid 1: Performing ([0-9]+) iterations.*/\1/p' "$LOG" | tail -1)
LAST=$(grep -E "^ +[0-9]+ +[0-9]+ +[0-9]+ +\*?[A-Z]" "$LOG" | tail -1)

echo "log    : $LOG"
echo "status : $STATE"
[ -n "$LAST" ] && echo "$LAST" | awk -v n="$NCYC" '{
    printf "iter   : %s / %s   (%.1f%%)   type %s\n", $2, n, (n?100*$2/n:0), $4
    printf "res rho: %.3e    res nuturb: %.3e\n", $8, $9
    printf "totalRes %.4e     Y+_max %.4f\n", $10, $11 }'

# rate and ETA, from the log's own mtime against the process start
if [ -n "$PID" ] && [ -n "$LAST" ]; then
    SEC=$(ps -o etimes= -p "$PID" | tr -d ' ')
    IT=$(echo "$LAST" | awk '{print $2}')
    awk -v s="$SEC" -v i="$IT" -v n="$NCYC" 'BEGIN{
        if (i>0 && s>0) { r=i/s; printf "rate   : %.1f iter/s", r
            if (n>i) printf "   ETA to %d cycles: %d min", n, (n-i)/r/60
            print "" } }'
fi
echo "tail -f: tail -f $LOG | grep --line-buffered -E '^ +[0-9]+ +[0-9]+'"
