run_cmd() {
    local cmd="$1"
    local outfile="$2"

    if [ -n "$outfile" ]; then
        echo "$cmd" >> "$outfile"
        eval "$cmd" >> "$outfile"
    else
        echo "$cmd"
        eval "$cmd"
    fi
}