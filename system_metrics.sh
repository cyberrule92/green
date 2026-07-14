#!/bin/bash

set -u

CPU_TDP="${CPU_TDP:-125}"
GPU_TDP="${GPU_TDP:-350}"
CO2_FACTOR="${CO2_FACTOR:-0.475}"
WATTS_PER_GB="${WATTS_PER_GB:-15}"
DISABLE_GPU_METRICS="${DISABLE_GPU_METRICS:-1}"

normalize() {
    local value
    value="$(echo "${1:-}" | xargs 2>/dev/null || true)"
    if [[ -z "$value" || "$value" == "N/A" || "$value" == "[N/A]" ]]; then
        echo "0"
    else
        echo "$value"
    fi
}

round2() {
    if [[ "${1:-}" == "N/A" || "${1:-}" == "" ]]; then
        echo "0.00"
    else
        printf "%.2f" "$1"
    fi
}

is_number() {
    [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

safe_bc() {
    local expression="$1"
    echo "$expression" | bc -l 2>/dev/null || echo "0"
}

read_cpu_util() {
    local cpu_line idle_field
    cpu_line="$(top -bn1 2>/dev/null | grep "Cpu(s)" | head -n1 || true)"
    if [[ -n "$cpu_line" ]]; then
        idle_field="$(echo "$cpu_line" | awk '{print 100 - $8}' 2>/dev/null || true)"
        if is_number "$idle_field"; then
            echo "$idle_field"
            return
        fi
    fi
    echo "0"
}

cpu_util="$(read_cpu_util)"

gpu_util="0"
mem_total="0"
mem_used="0"
mem_free="0"
sm_clock="0"
mem_clock="0"
gr_clock="0"
video_clock="0"
temp_core="0"
power_draw="0"
pstate="unknown"

if [[ "$DISABLE_GPU_METRICS" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_output="$(
        nvidia-smi \
          --query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free,clocks.sm,clocks.mem,clocks.gr,clocks.video,temperature.gpu,power.draw,pstate \
          --format=csv,noheader,nounits 2>/dev/null | head -n1 || true
    )"

    if [[ -n "$gpu_output" ]]; then
        IFS=',' read -r _index _name gpu_util mem_total mem_used mem_free \
            sm_clock mem_clock gr_clock video_clock temp_core power_draw pstate <<< "$gpu_output"

        gpu_util="$(normalize "$gpu_util")"
        mem_total="$(normalize "$mem_total")"
        mem_used="$(normalize "$mem_used")"
        mem_free="$(normalize "$mem_free")"
        sm_clock="$(normalize "$sm_clock")"
        mem_clock="$(normalize "$mem_clock")"
        gr_clock="$(normalize "$gr_clock")"
        video_clock="$(normalize "$video_clock")"
        temp_core="$(normalize "$temp_core")"
        power_draw="$(normalize "$power_draw")"
        pstate="$(echo "${pstate:-unknown}" | xargs 2>/dev/null || echo "unknown")"
    fi
fi

if is_number "$cpu_util"; then
    cpu_power="$(safe_bc "$CPU_TDP * $cpu_util / 100")"
else
    cpu_power="0"
fi

if is_number "$power_draw" && [[ "$power_draw" != "0" ]]; then
    gpu_power="$power_draw"
elif is_number "$gpu_util" && [[ "$gpu_util" != "0" ]]; then
    gpu_power="$(safe_bc "$GPU_TDP * $gpu_util / 100")"
elif is_number "$mem_used" && [[ "$mem_used" != "0" ]]; then
    gpu_power="$(safe_bc "$mem_used / 1024 * $WATTS_PER_GB")"
else
    gpu_power="0"
fi

if is_number "$cpu_power" && is_number "$gpu_power"; then
    total_power="$(safe_bc "$cpu_power + $gpu_power")"
else
    total_power="$cpu_power"
fi

if is_number "$total_power"; then
    energy="$(safe_bc "$total_power / 3600")"
    co2="$(safe_bc "$energy * $CO2_FACTOR")"
else
    energy="0"
    co2="0"
fi

cat <<JSON
{"system_gpu_utilization":$(round2 "$gpu_util"),"system_gpu_TotalMemory":$(round2 "$mem_total"),"system_gpu_UsedMemory":$(round2 "$mem_used"),"system_gpu_MemoryFree":$(round2 "$mem_free"),"system_gpu_SMClockFrequency":$(round2 "$sm_clock"),"system_gpu_MemClockFrequency":$(round2 "$mem_clock"),"system_gpu_GraphicsClock":$(round2 "$gr_clock"),"system_gpu_VideoClock":$(round2 "$video_clock"),"system_gpu_CoreTemperature":$(round2 "$temp_core"),"system_gpu_PowerDraw":$(round2 "$gpu_power"),"system_gpu_PerformanceState":"$pstate","system_cpu_utilization":$(round2 "$cpu_util"),"system_cpu_power":$(round2 "$cpu_power"),"system_total_power":$(round2 "$total_power"),"system_energy":$(round2 "$energy"),"system_co2_emission":$(round2 "$co2")}
JSON
