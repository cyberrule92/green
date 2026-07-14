/**
 * CarbonDashboard — Adaptive Green AI
 * Displays real-time sustainability metrics from the new paper-aligned endpoints:
 * - Model Zoo status (Section 3.3)
 * - LLMCarbon operational + embodied carbon breakdown (Section 3.4.2, 4.1)
 * - MoE expert health (Section 5.3)
 * - Deferred queue status (Section 3.5.2)
 * - Grid carbon signals across zones (Section 3.5.3)
 * - Policy suggestion from RL foundation (Section 3.6.2)
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  DataTable,
  Meter,
  Spinner,
  Text,
  Tip,
} from 'grommet';
import {
  Alert,
  Checkmark,
  CloudUpload,
  Cpu,
  Performance,
  TreeOption,
  StatusCritical,
  StatusGood,
  StatusWarning,
} from 'grommet-icons';
import {
  dispatchQueueNow,
  fetchGridForecast,
  fetchGridZones,
  fetchModelZoo,
  fetchPolicySuggestion,
  fetchQueueStatus,
} from '../lib/api';

const REFRESH_MS = 30_000;
const FORECAST_THRESHOLD = 450;   // gCO₂/kWh — EcoServe deferral trigger

// ── Grid-carbon forecast chart ──────────────────────────────────────────────
// Draws the next 48 h of grid carbon intensity from /api/grid/forecast. When the
// provider returns no forecast points (only a real-time signal), the curve is
// anchored to the live reading and the shape is modeled — labelled honestly.
const CARBON_LOW = [1, 169, 130];    // #01A982  clean grid
const CARBON_MID = [230, 168, 23];   // #E6A817  rising
const CARBON_HIGH = [201, 64, 64];   // #C94040  peak

function mix(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}
function scaleColor(v) {
  const LO = 360, MD = 470, HI = 640;
  const c = v <= MD
    ? mix(CARBON_LOW, CARBON_MID, Math.min(1, Math.max(0, (v - LO) / (MD - LO))))
    : mix(CARBON_MID, CARBON_HIGH, Math.min(1, Math.max(0, (v - MD) / (HI - MD))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
// Deterministic modeled fallback: solar trough midday, coal/gas evening ramp.
function buildModeled() {
  const pts = [];
  let seed = 20260710;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const start = new Date().getHours();
  for (let i = 0; i <= 48; i++) {
    const h = (start + i) % 24;
    const solar = Math.max(0, Math.sin((h - 6) / 12 * Math.PI));
    let base = 520 - solar * 190 + Math.max(0, Math.cos((h - 20) / 24 * 2 * Math.PI)) * 40;
    if (h >= 18 && h <= 23) base += 150 * Math.exp(-Math.pow(h - 20.5, 2) / 3.2);
    if (h >= 0 && h <= 4) base -= 40;
    base += (rnd() - 0.5) * 26;
    pts.push({ v: Math.max(300, Math.min(720, base)) });
  }
  return pts;
}
function interpAt(points, k) {
  if (k <= points[0].h) return points[0].v;
  if (k >= points[points.length - 1].h) return points[points.length - 1].v;
  for (let i = 1; i < points.length; i++) {
    if (points[i].h >= k) {
      const a = points[i - 1], b = points[i], t = (k - a.h) / (b.h - a.h || 1);
      return a.v + (b.v - a.v) * t;
    }
  }
  return points[points.length - 1].v;
}

function drawForecast(canvas, series, liveVal) {
  const ctx = canvas.getContext('2d');
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const W = canvas.clientWidth || 600, H = canvas.clientHeight || 260;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const padL = 44, padR = 12, padT = 16, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  let lo = Infinity, hi = -Infinity;
  series.forEach((p) => { if (p.v < lo) lo = p.v; if (p.v > hi) hi = p.v; });
  lo = Math.min(lo, FORECAST_THRESHOLD - 40); hi = Math.max(hi, FORECAST_THRESHOLD + 40);
  lo = Math.max(0, Math.floor((lo - 20) / 50) * 50); hi = Math.ceil((hi + 20) / 50) * 50;
  if (hi - lo < 150) hi = lo + 150;

  const txt = getComputedStyle(canvas).color || 'rgb(90,100,95)';
  const x = (i) => padL + (i / 48) * plotW;
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

  const span = hi - lo;
  const step = span > 500 ? 150 : span > 300 ? 100 : 50;
  ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
  ctx.textBaseline = 'middle';
  for (let g = Math.ceil(lo / step) * step; g <= hi; g += step) {
    const gy = y(g);
    ctx.strokeStyle = txt; ctx.globalAlpha = 0.12; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(W - padR, gy); ctx.stroke();
    ctx.globalAlpha = 0.55; ctx.fillStyle = txt; ctx.textAlign = 'right';
    ctx.fillText(g, padL - 8, gy);
  }
  ctx.globalAlpha = 1;

  for (let i = 0; i < 48; i++) {
    if (series[i].v < FORECAST_THRESHOLD && series[i + 1].v < FORECAST_THRESHOLD) {
      ctx.fillStyle = 'rgba(1,169,130,0.10)';
      ctx.fillRect(x(i), padT, x(i + 1) - x(i) + 0.5, plotH);
    }
  }

  ctx.beginPath(); ctx.moveTo(x(0), y(series[0].v));
  for (let a = 1; a <= 48; a++) ctx.lineTo(x(a), y(series[a].v));
  ctx.lineTo(x(48), padT + plotH); ctx.lineTo(x(0), padT + plotH); ctx.closePath();
  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  grad.addColorStop(0, 'rgba(230,168,23,0.14)'); grad.addColorStop(1, 'rgba(1,169,130,0.02)');
  ctx.fillStyle = grad; ctx.fill();

  ctx.setLineDash([5, 5]); ctx.strokeStyle = 'rgb(201,64,64)'; ctx.globalAlpha = 0.75; ctx.lineWidth = 1.25;
  ctx.beginPath(); ctx.moveTo(padL, y(FORECAST_THRESHOLD)); ctx.lineTo(W - padR, y(FORECAST_THRESHOLD)); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  ctx.fillStyle = 'rgb(201,64,64)'; ctx.textAlign = 'left';
  ctx.fillText('defer › 450', padL + 6, y(FORECAST_THRESHOLD) - 8);

  ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  for (let s = 0; s < 48; s++) {
    ctx.strokeStyle = scaleColor((series[s].v + series[s + 1].v) / 2);
    ctx.beginPath(); ctx.moveTo(x(s), y(series[s].v)); ctx.lineTo(x(s + 1), y(series[s + 1].v)); ctx.stroke();
  }

  ctx.fillStyle = scaleColor(series[0].v);
  ctx.beginPath(); ctx.arc(x(0), y(series[0].v), 4, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = txt; ctx.globalAlpha = 0.6; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.font = '9.5px ui-monospace, Menlo, Consolas, monospace';
  const startHour = new Date().getHours();
  for (let t = 0; t <= 48; t += 6) {
    const hr = (startHour + t) % 24;
    ctx.fillText(t === 0 ? 'now' : (hr < 10 ? '0' + hr : hr) + ':00', x(t), padT + plotH + 7);
  }
  ctx.globalAlpha = 1;
}

function ForecastChart({ forecast, liveCI }) {
  const ref = React.useRef(null);
  const [mode, setMode] = useState('modeled');

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;

    const modeled = buildModeled();
    let series = modeled, m = 'modeled';
    const pts = (Array.isArray(forecast) ? forecast : [])
      .map((p) => ({ h: (p.timestamp * 1000 - Date.now()) / 3600000, v: +p.carbon_intensity }))
      .filter((p) => Number.isFinite(p.h) && Number.isFinite(p.v) && p.h >= -1.5)
      .sort((a, b) => a.h - b.h);

    if (pts.length >= 6) {
      series = [];
      for (let k = 0; k <= 48; k++) series.push({ v: interpAt(pts, k) });
      m = 'live';
    } else if (liveCI != null && Number.isFinite(liveCI)) {
      const delta = liveCI - modeled[0].v;
      series = modeled.map((p, i) => ({ v: Math.max(280, p.v + delta * Math.max(0, 1 - i / 12)) }));
      m = 'anchored';
    }
    setMode(m);

    const liveVal = (liveCI != null && Number.isFinite(liveCI)) ? liveCI : series[0].v;
    const draw = () => drawForecast(canvas, series, liveVal);
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [forecast, liveCI]);

  const badge = mode === 'live'
    ? { color: 'status-ok', label: 'live · forecast' }
    : mode === 'anchored'
      ? { color: 'status-warning', label: 'live now · modeled 48h' }
      : { color: 'dark-3', label: 'modeled' };

  return (
    <MetricCard
      title="🌱 Grid Carbon Forecast · next 48 h"
      action={<Badge color={badge.color} label={badge.label} />}
    >
      <canvas
        ref={ref}
        style={{ width: '100%', height: '260px', display: 'block' }}
        role="img"
        aria-label="Forecast of grid carbon intensity over the next 48 hours, with EcoServe dispatch windows below the 450 gCO2/kWh deferral threshold highlighted."
      />
      <Box direction="row" gap="medium" wrap margin={{ top: 'xsmall' }}>
        {[['#01a982', 'clean'], ['#e6a817', 'rising'], ['#c94040', 'peak']].map(([c, l]) => (
          <Box key={l} direction="row" align="center" gap="xxsmall">
            <Box width="10px" height="10px" round="2px" background={c} />
            <Text size="xsmall" color="text-soft">{l}</Text>
          </Box>
        ))}
      </Box>
      <Text size="xsmall" color="text-soft">
        Requests above 450 gCO₂/kWh defer into the next window under the line (shaded green).
        When the provider returns no forecast points, the shape is modeled and anchored to the live reading.
      </Text>
    </MetricCard>
  );
}

function Badge({ color, label, icon }) {
  return (
    <Box
      direction="row"
      align="center"
      gap="xsmall"
      pad={{ horizontal: 'small', vertical: 'xxsmall' }}
      background={{ color, opacity: 'medium' }}
      round="small"
    >
      {icon}
      <Text size="xsmall" weight="bold" color={color}>
        {label}
      </Text>
    </Box>
  );
}

function MetricCard({ title, children, action }) {
  return (
    <Card background="white" elevation="small" round="small" style={{ minWidth: '280px' }}>
      <CardHeader pad={{ horizontal: 'medium', vertical: 'small' }} background="#f4f8f7">
        <Box direction="row" align="center" justify="between" fill="horizontal">
          <Text weight="bold" size="small" color="text-strong">
            {title}
          </Text>
          {action}
        </Box>
      </CardHeader>
      <CardBody pad="medium" gap="small">
        {children}
      </CardBody>
    </Card>
  );
}

function CarbonBar({ value, max = 600, label }) {
  const pct = Math.min(value / max, 1);
  const color = pct < 0.4 ? 'status-ok' : pct < 0.7 ? 'status-warning' : 'status-critical';
  return (
    <Box gap="xxsmall">
      <Box direction="row" justify="between">
        <Text size="xsmall" color="text-soft">
          {label}
        </Text>
        <Text size="xsmall" weight="bold">
          {value?.toFixed(0)} gCO₂/kWh
        </Text>
      </Box>
      <Meter value={pct * 100} max={100} size="small" thickness="small" color={color} />
    </Box>
  );
}

export function CarbonDashboard() {
  const [zoo, setZoo] = useState(null);
  const [zones, setZones] = useState(null);
  const [queue, setQueue] = useState(null);
  const [policy, setPolicy] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [dispatching, setDispatching] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [zooData, zoneData, queueData, policyData, forecastData] = await Promise.allSettled([
        fetchModelZoo(),
        fetchGridZones(),
        fetchQueueStatus(),
        fetchPolicySuggestion({ lookbackEntries: 100 }),
        fetchGridForecast(),
      ]);
      if (zooData.status === 'fulfilled') setZoo(zooData.value);
      if (zoneData.status === 'fulfilled') setZones(zoneData.value);
      if (queueData.status === 'fulfilled') setQueue(queueData.value);
      if (policyData.status === 'fulfilled') setPolicy(policyData.value);
      if (forecastData.status === 'fulfilled') setForecast(forecastData.value?.forecast || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, REFRESH_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const handleDispatchNow = async () => {
    setDispatching(true);
    try {
      await dispatchQueueNow();
      await refresh();
    } finally {
      setDispatching(false);
    }
  };

  if (loading) {
    return (
      <Box align="center" justify="center" pad="large">
        <Spinner size="medium" />
        <Text size="small" color="text-soft" margin={{ top: 'small' }}>
          Loading sustainability dashboard…
        </Text>
      </Box>
    );
  }

  return (
    <Box pad="medium" gap="medium" overflow={{ vertical: 'auto' }}>
      {error && (
        <Box
          background={{ color: 'status-critical', opacity: 'weak' }}
          pad="small"
          round="small"
          direction="row"
          gap="small"
          align="center"
        >
          <Alert color="status-critical" size="small" />
          <Text size="small" color="status-critical">
            {error}
          </Text>
        </Box>
      )}

      {/* ── Grid Carbon (Section 3.5.3) ── */}
      <MetricCard title="⚡ Grid Carbon Signals">
        {zones ? (
          <Box gap="small">
            {Object.entries(zones.carbon_map || {}).map(([zone, ci]) => (
              <CarbonBar key={zone} value={ci} label={zone} />
            ))}
            {!zones.multi_region_enabled && (
              <Text size="xsmall" color="text-soft">
                Set MULTI_REGION_ENABLED=true for multi-region routing (Section 3.5.3)
              </Text>
            )}
          </Box>
        ) : (
          <Text size="small" color="text-soft">No grid signal data</Text>
        )}
      </MetricCard>

      {/* ── Grid Carbon Forecast (Section 3.5.2 / 3.2.2) ── */}
      <ForecastChart
        forecast={forecast}
        liveCI={zones ? (zones.carbon_map?.[zones.primary_zone]
          ?? Object.values(zones.carbon_map || {})[0] ?? null) : null}
      />

      {/* ── Model Zoo (Section 3.3) ── */}
      <MetricCard title="🗄️ Model Zoo Registry">
        {zoo ? (
          <Box gap="small">
            <Box direction="row" gap="small" wrap>
              <Badge color="accent" label={`${zoo.zoo?.total_models} models`} icon={<Performance size="small" />} />
              <Badge color="status-ok" label={`${zoo.zoo?.available_models} available`} icon={<StatusGood size="small" />} />
              {zoo.zoo?.moe_models > 0 && (
                <Badge color="brand" label={`${zoo.zoo?.moe_models} MoE`} icon={<Cpu size="small" />} />
              )}
            </Box>
            <Text size="xsmall" color="text-soft">
              Version: {zoo.zoo?.version} · Refresh: {zoo.zoo?.refresh_policy}
            </Text>
            <Box gap="xsmall">
              {(zoo.models || [])
                .filter((m) => m.available)
                .map((m) => (
                  <Box
                    key={m.id}
                    direction="row"
                    align="center"
                    justify="between"
                    pad={{ horizontal: 'small', vertical: 'xxsmall' }}
                    background="#f4f8f7"
                    round="xsmall"
                  >
                    <Box>
                      <Text size="xsmall" weight="bold">
                        {m.model_variant} {m.moe ? '(MoE)' : ''}
                      </Text>
                      <Text size="xsmall" color="text-soft">
                        {m.region_label || m.region} · {m.hardware} · HE={m.hardware_efficiency}
                      </Text>
                    </Box>
                    <Box align="end">
                      <Text size="xsmall" color="dark-3">
                        {(m.flop_count_per_token / 1e9).toFixed(1)}B FLOPs/tok
                      </Text>
                      <Text size="xsmall" color="text-soft">
                        PUE {m.pue} · {m.mfg_carbon_kg}kg mfg
                      </Text>
                    </Box>
                  </Box>
                ))}
            </Box>
          </Box>
        ) : (
          <Text size="small" color="text-soft">Model Zoo unavailable</Text>
        )}
      </MetricCard>

      {/* ── Deferred Queue (Section 3.5.2) ── */}
      <MetricCard
        title="⏱ Deferred Execution Queue"
        action={
          <Button
            size="small"
            label={dispatching ? '…' : 'Dispatch Now'}
            onClick={handleDispatchNow}
            disabled={dispatching}
            style={{ fontSize: '11px' }}
          />
        }
      >
        {queue ? (
          <Box gap="small">
            <Box direction="row" gap="small" wrap>
              <Badge
                color={queue.queue?.queue_size > 0 ? 'status-warning' : 'status-ok'}
                label={`${queue.queue?.queue_size ?? 0} pending`}
                icon={<CloudUpload size="small" />}
              />
              <Badge color="dark-3" label={`${queue.queue?.dispatched_total ?? 0} dispatched`} icon={<Checkmark size="small" />} />
            </Box>
            <Box direction="row" justify="between">
              <Text size="xsmall" color="text-soft">Carbon threshold</Text>
              <Text size="xsmall">{queue.queue?.high_carbon_threshold} gCO₂/kWh</Text>
            </Box>
            <Box direction="row" justify="between">
              <Text size="xsmall" color="text-soft">Current grid carbon</Text>
              <Text size="xsmall" weight="bold">
                {queue.queue?.current_carbon_g_per_kwh?.toFixed(0)} gCO₂/kWh
              </Text>
            </Box>
            {queue.queue?.queue_size > 0 && (
              <Box gap="xxsmall">
                <Text size="xsmall" weight="bold" color="text-soft">
                  Pending requests:
                </Text>
                {queue.queue.pending_requests.map((r) => (
                  <Box
                    key={r.request_id}
                    pad={{ horizontal: 'xsmall', vertical: 'xxsmall' }}
                    background="#fff8e1"
                    round="xsmall"
                  >
                    <Text size="xsmall">
                      {r.request_id.slice(0, 8)}… · deadline in {r.seconds_until_deadline.toFixed(0)}s
                    </Text>
                  </Box>
                ))}
              </Box>
            )}
          </Box>
        ) : (
          <Text size="small" color="text-soft">Queue unavailable</Text>
        )}
      </MetricCard>

      {/* ── Policy Suggestion / RL Foundation (Section 3.6.2) ── */}
      <MetricCard title="🧠 Policy Intelligence (RL Foundation)">
        {policy && policy.status !== 'insufficient_data' ? (
          <Box gap="small">
            <Box direction="row" gap="small" wrap>
              <Badge
                color={policy.observed?.sla_violation_rate > 0.1 ? 'status-critical' : 'status-ok'}
                label={`SLA violations: ${(policy.observed?.sla_violation_rate * 100).toFixed(1)}%`}
                icon={policy.observed?.sla_violation_rate > 0.1 ? <StatusCritical size="small" /> : <StatusGood size="small" />}
              />
              <Badge
                color="accent"
                label={`Avg CSS: ${policy.observed?.avg_css_score?.toFixed(3)}`}
                icon={<TreeOption size="small" />}
              />
            </Box>
            <Box>
              <Text size="xsmall" color="text-soft" margin={{ bottom: 'xxsmall' }}>
                Carbon (avg per request):
              </Text>
              <Text size="xsmall" weight="bold">
                {(policy.observed?.avg_carbon_g * 1e6).toFixed(3)} µgCO₂
              </Text>
            </Box>
            <Box gap="xxsmall">
              <Text size="xsmall" weight="bold" color="text-soft">
                Suggestions ({policy.entries_analyzed} entries analyzed):
              </Text>
              {policy.suggestions?.map((s, i) => (
                <Box
                  key={i}
                  pad={{ horizontal: 'small', vertical: 'xxsmall' }}
                  background="#e8f5e9"
                  round="xsmall"
                >
                  <Text size="xsmall">{s}</Text>
                </Box>
              ))}
            </Box>
            <Box gap="xxsmall">
              <Text size="xsmall" color="text-soft">Current policy coefficients:</Text>
              <Box direction="row" gap="xsmall" wrap>
                {Object.entries(policy.current_policy || {})
                  .filter(([k]) => ['carbon', 'latency', 'accuracy', 'cost'].includes(k))
                  .map(([k, v]) => (
                    <Box
                      key={k}
                      pad={{ horizontal: 'xsmall', vertical: 'xxsmall' }}
                      background="#e3f2fd"
                      round="xsmall"
                    >
                      <Text size="xsmall">
                        w_{k}: {Number(v).toFixed(2)}
                      </Text>
                    </Box>
                  ))}
              </Box>
            </Box>
          </Box>
        ) : (
          <Box gap="xsmall">
            <Text size="small" color="text-soft">Insufficient decision history for policy analysis.</Text>
            <Text size="xsmall" color="text-soft">
              Policy suggestions appear after {policy?.entries_analyzed ?? 0} entries reach the audit log.
            </Text>
          </Box>
        )}
      </MetricCard>

      <Box align="end">
        <Text size="xsmall" color="text-soft">
          Auto-refreshes every {REFRESH_MS / 1000}s · Adaptive Green AI v4.0
        </Text>
      </Box>
    </Box>
  );
}
