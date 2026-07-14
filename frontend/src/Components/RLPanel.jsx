/**
 * RLPanel — Adaptive Green AI
 * Read-only display of the online REINFORCE policy controller state.
 *
 * IMPORTANT: There are no controls here. Weights are learned automatically
 * from real inference outcomes — no offline training, no user configuration.
 *
 * Shows per-tier:
 *  - Current learned coefficients (w_carbon, w_latency, w_accuracy, w_cost)
 *  - Reward trend sparkline (last 30 episodes)
 *  - Episode count and policy version
 *  - Exploration status (Dirichlet noise active/inactive)
 *  - Convergence indicator
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardBody,
  CardHeader,
  Meter,
  Spinner,
  Tab,
  Tabs,
  Text,
  Tip,
} from 'grommet';
import {
  Alert,
  CircleInformation,
  Refresh,
  StatusGood,
  StatusWarning,
  TreeOption,
} from 'grommet-icons';
import { fetchRLHistory, fetchRLStatus } from '../lib/api';

const TIER_COLORS = {
  standard: '#01a982',
  premium:  '#614767',
  esg:      '#2c7a3f',
  batch:    '#c57523',
};

const COEF_LABELS = {
  carbon:   { label: 'Carbon  (w_c)',   color: '#388e3c', desc: 'Lower carbon emissions' },
  latency:  { label: 'Latency (w_l)',   color: '#1565c0', desc: 'SLA conformance' },
  accuracy: { label: 'Accuracy (w_a)',  color: '#6a1b9a', desc: 'Response quality' },
  cost:     { label: 'Cost    (w_cost)', color: '#e65100', desc: 'Compute cost efficiency' },
};

const REFRESH_MS = 15_000;

// ── Mini sparkline ─────────────────────────────────────────────────────────
function Sparkline({ values, width = 160, height = 32 }) {
  if (!values || values.length < 2) {
    return <Text size="xsmall" color="text-soft">No data yet</Text>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 0.001;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const lastY = parseFloat(pts[pts.length - 1].split(',')[1]);
  const avgRaw = values.reduce((a, b) => a + b, 0) / values.length;

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke="#01a982"
        strokeWidth="1.5"
      />
      {/* Last value dot */}
      <circle
        cx={width}
        cy={lastY}
        r={2.5}
        fill="#01a982"
      />
      {/* Average baseline */}
      <line
        x1={0}
        y1={height - ((avgRaw - min) / range) * (height - 4) - 2}
        x2={width}
        y2={height - ((avgRaw - min) / range) * (height - 4) - 2}
        stroke="#aaa"
        strokeWidth="0.8"
        strokeDasharray="3,2"
      />
    </svg>
  );
}

// ── Weight bar ────────────────────────────────────────────────────────────
function WeightBar({ label, value, color, desc }) {
  return (
    <Tip content={<Text size="xsmall">{desc}</Text>} dropProps={{ align: { left: 'right' } }}>
      <Box gap="xxsmall">
        <Box direction="row" justify="between" align="center">
          <Text size="xsmall" color="dark-3" style={{ fontFamily: 'monospace' }}>
            {label}
          </Text>
          <Text size="xsmall" weight="bold" color={color}>
            {(value * 100).toFixed(1)}%
          </Text>
        </Box>
        <Meter
          value={value * 100}
          max={60}
          size="small"
          thickness="xsmall"
          color={color}
        />
      </Box>
    </Tip>
  );
}

// ── Trend badge ───────────────────────────────────────────────────────────
function TrendBadge({ trend }) {
  const map = {
    improving:         { icon: <StatusGood size="small" />,         color: 'status-ok',      label: 'Improving' },
    stable:            { icon: <CircleInformation size="small" />,  color: 'brand',           label: 'Stable' },
    declining:         { icon: <StatusWarning size="small" />,      color: 'status-warning',  label: 'Declining' },
    insufficient_data: { icon: <Refresh size="small" />,            color: 'dark-4',          label: 'Learning…' },
  };
  const { icon, color, label } = map[trend] || map.insufficient_data;
  return (
    <Box direction="row" align="center" gap="xxsmall">
      {React.cloneElement(icon, { color })}
      <Text size="xsmall" color={color}>{label}</Text>
    </Box>
  );
}

// ── Per-tier card ─────────────────────────────────────────────────────────
function TierCard({ tierName, tierData, history }) {
  const weights = tierData?.weights || {};
  const rewardHistory = (history || []).slice(-30);

  return (
    <Card background="white" elevation="small" round="small">
      <CardHeader
        pad={{ horizontal: 'medium', vertical: 'small' }}
        background={{ color: TIER_COLORS[tierName] || '#01a982', opacity: 'strong' }}
      >
        <Box direction="row" align="center" justify="between" fill="horizontal">
          <Box direction="row" align="center" gap="small">
            <TreeOption size="small" color="white" />
            <Text weight="bold" size="small" color="white" style={{ textTransform: 'capitalize' }}>
              {tierName}
            </Text>
            <Box
              pad={{ horizontal: 'xsmall', vertical: 'xxsmall' }}
              background={{ color: 'white', opacity: 'weak' }}
              round="xsmall"
            >
              <Text size="xsmall" color="white">
                v{tierData?.policy_version}
              </Text>
            </Box>
          </Box>
          <TrendBadge trend={tierData?.reward_trend} />
        </Box>
      </CardHeader>

      <CardBody pad="medium" gap="small">
        {/* Learned weights — read-only display */}
        <Box gap="xsmall">
          <Text size="xsmall" color="text-soft" weight="bold">
            LEARNED POLICY WEIGHTS
          </Text>
          <Box
            pad={{ horizontal: 'xsmall', vertical: 'xxsmall' }}
            background={{ color: 'status-warning', opacity: 'weak' }}
            round="xsmall"
            direction="row"
            align="center"
            gap="xsmall"
          >
            <Alert size="xsmall" color="status-warning" />
            <Text size="xsmall" color="dark-3">
              Auto-updated by online REINFORCE — read only
            </Text>
          </Box>
          <Box gap="xxsmall">
            {Object.entries(COEF_LABELS).map(([key, meta]) => (
              <WeightBar
                key={key}
                label={meta.label}
                value={weights[key] || 0}
                color={meta.color}
                desc={meta.desc}
              />
            ))}
          </Box>
        </Box>

        {/* Reward sparkline */}
        <Box gap="xxsmall">
          <Box direction="row" justify="between" align="center">
            <Text size="xsmall" color="text-soft" weight="bold">
              REWARD TREND (last {rewardHistory.length} episodes)
            </Text>
            <Text size="xsmall" color="text-soft">
              avg {(tierData?.recent_avg_reward ?? 0).toFixed(3)}
            </Text>
          </Box>
          <Sparkline values={rewardHistory} width={240} height={36} />
        </Box>

        {/* Stats row */}
        <Box direction="row" gap="small" wrap>
          <Box
            pad={{ horizontal: 'small', vertical: 'xxsmall' }}
            background="#f4f8f7"
            round="xsmall"
          >
            <Text size="xsmall" color="text-soft">Episodes</Text>
            <Text size="xsmall" weight="bold">{tierData?.episode_count ?? 0}</Text>
          </Box>
          <Box
            pad={{ horizontal: 'small', vertical: 'xxsmall' }}
            background="#f4f8f7"
            round="xsmall"
          >
            <Text size="xsmall" color="text-soft">Baseline EMA</Text>
            <Text size="xsmall" weight="bold">{(tierData?.baseline_ema ?? 0).toFixed(3)}</Text>
          </Box>
          <Box
            pad={{ horizontal: 'small', vertical: 'xxsmall' }}
            background={tierData?.episode_count > 0 ? '#e8f5e9' : '#f4f8f7'}
            round="xsmall"
          >
            <Text size="xsmall" color="text-soft">Learning</Text>
            <Text size="xsmall" weight="bold" color={tierData?.episode_count > 0 ? 'status-ok' : 'dark-4'}>
              {tierData?.episode_count > 0 ? 'Active' : 'Waiting…'}
            </Text>
          </Box>
        </Box>
      </CardBody>
    </Card>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────
export function RLPanel() {
  const [rl, setRl] = useState(null);
  const [history, setHistory] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);

  const TIERS = ['standard', 'premium', 'esg', 'batch'];

  const refresh = useCallback(async () => {
    try {
      const [statusRes, histRes] = await Promise.allSettled([
        fetchRLStatus(),
        fetchRLHistory({ lastN: 100 }),
      ]);
      if (statusRes.status === 'fulfilled') setRl(statusRes.value?.rl);
      if (histRes.status === 'fulfilled') setHistory(histRes.value?.history || {});
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

  if (loading) {
    return (
      <Box align="center" justify="center" pad="large">
        <Spinner size="medium" />
        <Text size="small" color="text-soft" margin={{ top: 'small' }}>
          Loading RL policy state…
        </Text>
      </Box>
    );
  }

  const globalStats = rl ? {
    totalEpisodes: TIERS.reduce((s, t) => s + (rl.tiers?.[t]?.episode_count ?? 0), 0),
    exploration: rl.exploration_enabled,
    alpha0: rl.alpha_0,
    wMin: rl.w_min,
  } : {};

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
          <Text size="small" color="status-critical">{error}</Text>
        </Box>
      )}

      {/* Global summary bar */}
      <Box
        background="white"
        pad={{ horizontal: 'medium', vertical: 'small' }}
        round="small"
        elevation="small"
        direction="row"
        gap="medium"
        wrap
        align="center"
        justify="between"
      >
        <Box>
          <Text size="small" weight="bold" color="text-strong">
            🧠 Online REINFORCE Policy Controller
          </Text>
          <Text size="xsmall" color="text-soft">
            Self-adapting from real inference outcomes · No offline training · No user controls
          </Text>
        </Box>
        <Box direction="row" gap="medium" wrap align="center">
          <Box align="center">
            <Text size="xsmall" color="text-soft">Total Episodes</Text>
            <Text size="small" weight="bold">{globalStats.totalEpisodes ?? 0}</Text>
          </Box>
          <Box align="center">
            <Text size="xsmall" color="text-soft">Learning Rate α₀</Text>
            <Text size="small" weight="bold">{globalStats.alpha0}</Text>
          </Box>
          <Box align="center">
            <Text size="xsmall" color="text-soft">Exploration</Text>
            <Text size="small" weight="bold" color={globalStats.exploration ? 'status-ok' : 'dark-4'}>
              {globalStats.exploration ? 'Dirichlet ON' : 'Off'}
            </Text>
          </Box>
          <Box align="center">
            <Text size="xsmall" color="text-soft">Weight Floor</Text>
            <Text size="small" weight="bold">{(globalStats.wMin * 100).toFixed(0)}%</Text>
          </Box>
        </Box>
      </Box>

      {/* Algorithm note */}
      <Box
        background={{ color: 'brand', opacity: 'weak' }}
        pad={{ horizontal: 'medium', vertical: 'small' }}
        round="small"
        direction="row"
        gap="small"
        align="start"
      >
        <CircleInformation size="small" color="brand" />
        <Box>
          <Text size="xsmall" weight="bold" color="brand">How it works</Text>
          <Text size="xsmall" color="dark-3">
            After each inference: reward R = 0.35·SLA + 0.30·carbon + 0.25·accuracy + 0.10·cost.
            Gradient ∇w_i = (R − EMA_baseline) × (score_i(selected) − E[score_i]).
            Weights updated with decaying lr α_t = α₀ / (1 + √t), then projected onto the simplex.
            Dirichlet noise provides exploration without UI intervention.
          </Text>
        </Box>
      </Box>

      {/* Per-tier tabs */}
      <Tabs activeIndex={activeTab} onActive={setActiveTab}>
        {TIERS.map((tier) => (
          <Tab key={tier} title={<Text size="small" style={{ textTransform: 'capitalize' }}>{tier}</Text>}>
            <Box pad={{ top: 'small' }}>
              <TierCard
                tierName={tier}
                tierData={rl?.tiers?.[tier]}
                history={history[tier]}
              />
            </Box>
          </Tab>
        ))}
      </Tabs>

      <Box align="end">
        <Text size="xsmall" color="text-soft">
          Auto-refreshes every {REFRESH_MS / 1000}s · Adaptive Green AI RL Controller
        </Text>
      </Box>
    </Box>
  );
}
