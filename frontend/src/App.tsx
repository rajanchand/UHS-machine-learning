import { useState, useEffect, useCallback, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ---- Types ----
interface StreamEvent {
  event_type: string;
  flow_id: string;
  ts: string;
  src_ip: string;
  dst_ip: string;
  protocol: number;
  score: number | null;
  is_anomaly: boolean | null;
  model_name: string | null;
  alert_id: string | null;
  severity: string | null;
  suspected_attack_type: string | null;
}

interface KPI {
  total_flows: number;
  total_alerts: number;
  open_alerts: number;
  estimated_fpr: number;
  top_talkers: { ip: string; flow_count: number }[];
}

interface TimelinePoint {
  timestamp: string;
  avg_score: number;
  max_score: number;
  flow_count: number;
  anomaly_count: number;
}

interface AlertItem {
  id: string;
  flow_id: string;
  severity: string;
  suspected_attack_type: string | null;
  status: string;
  created_at: string;
  feedback_verdict: string | null;
}

// ---- Hooks ----
function useSSE(url: string, onEvent?: (ev: StreamEvent) => void) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const onEventRef = useRef(onEvent);

  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    const eventSource = new EventSource(url, { withCredentials: true });

    eventSource.onopen = () => setConnected(true);
    eventSource.onerror = () => setConnected(false);

    eventSource.onmessage = (e) => {
      try {
        const data: StreamEvent = JSON.parse(e.data);
        setEvents(prev => [data, ...prev].slice(0, 200));
        if (onEventRef.current) {
          onEventRef.current(data);
        }
      } catch { /* ignore malformed events */ }
    };

    return () => {
      eventSource.close();
      setConnected(false);
    };
  }, [url]);

  return { events, connected };
}

// ---- Components ----
function LoginForm({ onLoginSuccess }: { onLoginSuccess: (username: string) => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ username, password }),
      });
      if (res.ok) {
        const data = await res.json();
        onLoginSuccess(data.username);
      } else {
        const err = await res.json();
        setError(err.detail || 'Invalid username or password');
      }
    } catch {
      setError('Connection to backend failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card glass-card">
        <div className="login-header">
          <span className="login-logo">🛡️</span>
          <h2>Network Anomaly Detection</h2>
          <p>Sign in to access the security console</p>
        </div>
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="e.g. analyst"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              required
            />
          </div>
          {error && <div className="login-error">⚠ {error}</div>}
          <button type="submit" className="login-btn" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
        <div className="login-tip">
          <strong>Quick Tip:</strong> Use the seeded credentials:<br />
          <code>analyst</code> / <code>password123</code>
        </div>
      </div>
    </div>
  );
}

function KPIStrip({
  kpi,
  threshold,
  loading,
  error,
}: {
  kpi: KPI | null;
  threshold: number;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return (
      <div className="kpi-strip">
        {[0, 1, 2, 3].map(i => (
          <div key={i} className="kpi-card">
            <div className="kpi-label">Loading KPIs...</div>
            <div className="kpi-value blue" style={{ display: 'flex', alignItems: 'center', height: '2rem' }}>
              <div className="loading-spinner" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="kpi-strip">
        <div className="kpi-card error-state" style={{ gridColumn: '1 / -1', margin: 0 }}>
          <span>⚠ {error}</span>
        </div>
      </div>
    );
  }

  if (!kpi) {
    return (
      <div className="kpi-strip">
        {[0, 1, 2, 3].map(i => (
          <div key={i} className="kpi-card">
            <div className="kpi-label">No data</div>
            <div className="kpi-value blue">—</div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="kpi-strip">
      <div className="kpi-card">
        <span className="kpi-label">Flows Processed</span>
        <span className="kpi-value blue">{kpi.total_flows.toLocaleString()}</span>
        <span className="kpi-subtitle">Total ingested</span>
      </div>
      <div className="kpi-card">
        <span className="kpi-label">Alerts Raised</span>
        <span className="kpi-value red">{kpi.total_alerts.toLocaleString()}</span>
        <span className="kpi-subtitle">{kpi.open_alerts} open</span>
      </div>
      <div className="kpi-card">
        <span className="kpi-label">Estimated FPR</span>
        <span className="kpi-value amber">{(kpi.estimated_fpr * 100).toFixed(2)}%</span>
        <span className="kpi-subtitle">At threshold {threshold.toFixed(2)}</span>
      </div>
      <div className="kpi-card">
        <span className="kpi-label">Top Talkers</span>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
          {kpi.top_talkers.length > 0 ? kpi.top_talkers.slice(0, 3).map(t => (
            <span key={t.ip} className="ip-badge">{t.ip}</span>
          )) : <span className="kpi-subtitle">No data yet</span>}
        </div>
      </div>
    </div>
  );
}

function TimelineChart({
  data,
  threshold,
  loading,
  error,
}: {
  data: TimelinePoint[];
  threshold: number;
  loading: boolean;
  error: string | null;
}) {
  const formatted = data.map(p => ({
    ...p,
    time: new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }));

  return (
    <div className="glass-card chart-container" id="anomaly-timeline">
      <div className="glass-card-header">
        <span className="glass-card-title">Anomaly Score Timeline</span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {data.length > 0 ? `${data.length} data points` : 'Waiting for data...'}
        </span>
      </div>
      {loading ? (
        <div className="empty-state">
          <div className="loading-spinner" />
          <div>Loading timeline metrics...</div>
        </div>
      ) : error ? (
        <div className="empty-state error-state">
          <div>⚠ {error}</div>
        </div>
      ) : data.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📊</div>
          <div>No timeline data yet. Start the simulator to see live scores.</div>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={formatted} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="maxGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis domain={[0, 1]} />
            <Tooltip
              contentStyle={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-card)',
                borderRadius: 'var(--radius-md)',
                fontSize: '0.8125rem',
              }}
            />
            <Legend />
            <ReferenceLine
              y={threshold}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{ value: `Threshold: ${threshold.toFixed(2)}`, fill: '#f59e0b', fontSize: 11 }}
            />
            <Area
              type="monotone" dataKey="avg_score" name="Avg Score"
              stroke="#3b82f6" fill="url(#scoreGradient)" strokeWidth={2}
            />
            <Area
              type="monotone" dataKey="max_score" name="Max Score"
              stroke="#ef4444" fill="url(#maxGradient)" strokeWidth={1.5}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

function FlowFeed({ events }: { events: StreamEvent[] }) {
  const [paused, setPaused] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!paused && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events, paused]);

  return (
    <div className="glass-card" id="flow-feed">
      <div className="glass-card-header">
        <span className="glass-card-title">Live Flow Feed</span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {events.length} flows
        </span>
      </div>
      {events.length === 0 ? (
        <div className="empty-state">
          <div className="loading-spinner" />
          <div>Waiting for flows...</div>
        </div>
      ) : (
        <div
          className="table-scroll"
          ref={scrollRef}
          onMouseEnter={() => setPaused(true)}
          onMouseLeave={() => setPaused(false)}
        >
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Source → Dest</th>
                <th>Proto</th>
                <th>Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev, i) => (
                <tr key={`${ev.flow_id}-${i}`}>
                  <td>{new Date(ev.ts).toLocaleTimeString()}</td>
                  <td>
                    <span className="ip-badge">{ev.src_ip}</span>
                    {' → '}
                    <span className="ip-badge">{ev.dst_ip}</span>
                  </td>
                  <td>{ev.protocol === 6 ? 'TCP' : ev.protocol === 17 ? 'UDP' : ev.protocol}</td>
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {ev.score !== null ? ev.score.toFixed(3) : '—'}
                  </td>
                  <td>
                    <span className={`badge ${ev.is_anomaly ? 'anomaly' : 'normal'}`}>
                      {ev.is_anomaly ? '⚠ Anomaly' : '✓ Normal'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function SystemOperations({
  drift,
  driftLoading,
  driftError,
  activeScenario,
  onSimulate,
}: {
  drift: any;
  driftLoading: boolean;
  driftError: string | null;
  activeScenario: string | null;
  onSimulate: (scenario: string | null) => void;
}) {
  const [showFeatures, setShowFeatures] = useState(false);

  return (
    <div className="glass-card" id="system-operations">
      <div className="glass-card-header">
        <span className="glass-card-title">System Operations</span>
      </div>
      
      {/* Simulation injection control panel */}
      <div className="op-section">
        <h4 className="op-section-title">Attack Scenario Injector</h4>
        <p className="op-section-desc">
          Inject real-time attack flows to check classifier response and alert rules.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
          <div className="scenario-buttons">
            <button
              className={`sim-btn ${activeScenario === 'ddos' ? 'active ddos' : ''}`}
              onClick={() => onSimulate('ddos')}
            >
              🔥 DDoS
            </button>
            <button
              className={`sim-btn ${activeScenario === 'port_scan' ? 'active scan' : ''}`}
              onClick={() => onSimulate('port_scan')}
            >
              🔎 Port Scan
            </button>
            <button
              className={`sim-btn ${activeScenario === 'brute_force' ? 'active brute' : ''}`}
              onClick={() => onSimulate('brute_force')}
            >
              ⚔ Brute Force
            </button>
          </div>
          {activeScenario ? (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
              <span className="scenario-status red" style={{ animation: 'pulse-glow 1.5s infinite' }}>
                ⚠ Replaying {activeScenario.replace(/_/g, ' ').toUpperCase()}...
              </span>
              <button className="sim-btn stop" onClick={() => onSimulate(null)}>
                Stop
              </button>
            </div>
          ) : (
            <span className="scenario-status green" style={{ marginTop: '4px' }}>
              ✓ Normal Traffic Ingestion
            </span>
          )}
        </div>
      </div>

      {/* Drift monitoring details */}
      <div className="op-section" style={{ marginTop: 'var(--space-lg)', borderTop: '1px solid var(--border-subtle)', paddingTop: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h4 className="op-section-title">Baseline Drift (PSI)</h4>
          {drift && drift.feature_psis && (
            <button className="close-btn" style={{ fontSize: '0.6875rem', padding: '2px 6px' }} onClick={() => setShowFeatures(!showFeatures)}>
              {showFeatures ? 'Hide Details' : 'Show Features'}
            </button>
          )}
        </div>
        
        {driftLoading ? (
          <div className="empty-state" style={{ padding: 'var(--space-md)' }}>
            <div className="loading-spinner" />
            <div>Calculating drift...</div>
          </div>
        ) : driftError ? (
          <div className="error-state">
            <span>⚠ {driftError}</span>
          </div>
        ) : drift ? (
          <div style={{ marginTop: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="status-label" style={{ textTransform: 'none', letterSpacing: 'normal' }}>Overall stability index:</span>
              <span className={`badge ${drift.status === 'drifting' ? 'anomaly' : 'normal'}`} style={{ fontSize: '0.75rem' }}>
                {drift.overall_psi.toFixed(4)} ({drift.status.toUpperCase()})
              </span>
            </div>
            {drift.status === 'drifting' && (
              <div className="stale-alert-banner">
                <strong>Model is stale!</strong> The traffic distribution has drifted from training baseline. Retraining is recommended.
              </div>
            )}
            
            {showFeatures && drift.feature_psis && (
              <div className="feature-psi-container" style={{ marginTop: '10px' }}>
                <div className="feature-psi-header">
                  <span>Feature</span>
                  <span>PSI Value</span>
                </div>
                <div className="feature-psi-scroll">
                  {Object.entries(drift.feature_psis).map(([feat, val]: [string, any]) => (
                    <div key={feat} className="feature-psi-row">
                      <span className="feat-name">{feat}</span>
                      <span className={`feat-val ${val >= 0.25 ? 'drifting' : 'stable'}`}>
                        {val.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="empty-state" style={{ padding: 'var(--space-md)' }}>
            <span>No drift measurements yet</span>
          </div>
        )}
      </div>
    </div>
  );
}

function AlertTable({
  alerts,
  onSelect,
  loading,
  error,
  onExportCSV,
}: {
  alerts: AlertItem[];
  onSelect: (alert: AlertItem) => void;
  loading: boolean;
  error: string | null;
  onExportCSV: () => void;
}) {
  const [sortField, setSortField] = useState<'created_at' | 'severity'>('created_at');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const severityOrder: Record<string, number> = { critical: 4, high: 3, medium: 2, low: 1 };

  const filtered = alerts.filter(a => filterStatus === 'all' || a.status === filterStatus);
  const sorted = [...filtered].sort((a, b) => {
    if (sortField === 'severity') {
      return (severityOrder[b.severity] || 0) - (severityOrder[a.severity] || 0);
    }
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

  return (
    <div className="glass-card" id="alert-table">
      <div className="glass-card-header">
        <span className="glass-card-title">Alerts</span>
        <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center' }}>
          <button onClick={onExportCSV} className="close-btn" style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
            📥 Export Feedback
          </button>
          <select
            className="model-select"
            value={filterStatus}
            onChange={e => setFilterStatus(e.target.value)}
          >
            <option value="all">All Status</option>
            <option value="open">Open</option>
            <option value="acknowledged">Acknowledged</option>
            <option value="resolved">Resolved</option>
          </select>
          <select
            className="model-select"
            value={sortField}
            onChange={e => setSortField(e.target.value as 'created_at' | 'severity')}
          >
            <option value="created_at">Sort: Time</option>
            <option value="severity">Sort: Severity</option>
          </select>
        </div>
      </div>
      {loading ? (
        <div className="empty-state">
          <div className="loading-spinner" />
          <div>Loading alerts...</div>
        </div>
      ) : error ? (
        <div className="empty-state error-state">
          <div>⚠ {error}</div>
        </div>
      ) : sorted.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">🛡️</div>
          <div>{alerts.length === 0 ? 'No alerts yet' : 'No alerts match filters'}</div>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Severity</th>
                <th>Attack Type</th>
                <th>Verdict</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(alert => (
                <tr key={alert.id} onClick={() => onSelect(alert)}>
                  <td>{new Date(alert.created_at).toLocaleTimeString()}</td>
                  <td><span className={`badge ${alert.severity}`}>{alert.severity}</span></td>
                  <td>{alert.suspected_attack_type || 'Unknown'}</td>
                  <td>
                    {alert.feedback_verdict ? (
                      <span className={`badge ${alert.feedback_verdict === 'true_positive' ? 'tp' : 'fp'}`}>
                        {alert.feedback_verdict === 'true_positive' ? 'True Pos' : 'False Pos'}
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Pending</span>
                    )}
                  </td>
                  <td><span className={`badge ${alert.status}`}>{alert.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function DrillDown({
  alert,
  onClose,
  onSubmitFeedback,
}: {
  alert: AlertItem;
  onClose: () => void;
  onSubmitFeedback: (alertId: string, verdict: string) => void;
}) {
  return (
    <div className="drill-down-overlay" onClick={onClose}>
      <div className="drill-down-panel" onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2>Alert Details</h2>
          <button className="close-btn" onClick={onClose}>✕ Close</button>
        </div>
        <div style={{ marginTop: 'var(--space-lg)' }}>
          <div className="feature-grid">
            <span className="feature-label">Alert ID</span>
            <span className="feature-value" style={{ fontSize: '0.75rem' }}>{alert.id.slice(0, 8)}...</span>

            <span className="feature-label">Flow ID</span>
            <span className="feature-value" style={{ fontSize: '0.75rem' }}>{alert.flow_id.slice(0, 8)}...</span>

            <span className="feature-label">Severity</span>
            <span className="feature-value"><span className={`badge ${alert.severity}`}>{alert.severity}</span></span>

            <span className="feature-label">Attack Type</span>
            <span className="feature-value">{alert.suspected_attack_type || 'Unknown'}</span>

            <span className="feature-label">Status</span>
            <span className="feature-value"><span className={`badge ${alert.status}`}>{alert.status}</span></span>

            <span className="feature-label">Created</span>
            <span className="feature-value">{new Date(alert.created_at).toLocaleString()}</span>
          </div>

          <div style={{ marginTop: 'var(--space-xl)', borderTop: '1px solid var(--border-subtle)', paddingTop: 'var(--space-lg)' }}>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 'var(--space-md)' }}>
              ANALYST FEEDBACK LOOP
            </h3>
            {alert.feedback_verdict ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span className="feature-label">Submitted Verdict:</span>
                  <span className={`badge ${alert.feedback_verdict === 'true_positive' ? 'tp' : 'fp'}`}>
                    {alert.feedback_verdict === 'true_positive' ? '✓ True Positive' : '✗ False Positive'}
                  </span>
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                  Verdicts are stored locally and exported in retrain-ready dataset format.
                </p>
                <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                  <button
                    className="close-btn"
                    style={{ fontSize: '0.75rem' }}
                    onClick={() => onSubmitFeedback(alert.id, alert.feedback_verdict === 'true_positive' ? 'false_positive' : 'true_positive')}
                  >
                    Change Verdict
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                  Verify if this prediction is a real attack or a normal traffic variance (false alarm).
                </p>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <button
                    className="feedback-btn tp"
                    onClick={() => onSubmitFeedback(alert.id, 'true_positive')}
                  >
                    ✓ True Positive
                  </button>
                  <button
                    className="feedback-btn fp"
                    onClick={() => onSubmitFeedback(alert.id, 'false_positive')}
                  >
                    ✗ False Positive
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---- Main App ----
function App() {
  const [user, setUser] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  const [kpi, setKpi] = useState<KPI | null>(null);
  const [kpiLoading, setKpiLoading] = useState(false);
  const [kpiError, setKpiError] = useState<string | null>(null);

  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const [timelineError, setTimelineError] = useState<string | null>(null);

  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [alertsLoading, setAlertsLoading] = useState(false);
  const [alertsError, setAlertsError] = useState<string | null>(null);

  const [drift, setDrift] = useState<any>(null);
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftError, setDriftError] = useState<string | null>(null);

  const [selectedAlert, setSelectedAlert] = useState<AlertItem | null>(null);
  const [activeModel, setActiveModel] = useState('isolation_forest');
  const [threshold, setThreshold] = useState(0.5);
  const [models, setModels] = useState<string[]>(['isolation_forest', 'autoencoder', 'halfspace_trees', 'lightgbm_benchmark']);

  const [activeScenario, setActiveScenario] = useState<string | null>(null);
  
  // Real-time rates calculations
  const [flowsPerSec, setFlowsPerSec] = useState(0);
  const [alertRate, setAlertRate] = useState(0);
  const flowCounter = useRef(0);
  const alertCounter = useRef(0);

  // Check auth session
  const checkAuth = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/v1/auth/me`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setUser(data.username);
      } else {
        setUser(null);
      }
    } catch {
      setUser(null);
    } finally {
      setAuthLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  // Handle stream event callback for rate calculation
  const handleIncomingEvent = useCallback((ev: StreamEvent) => {
    flowCounter.current += 1;
    if (ev.is_anomaly) {
      alertCounter.current += 1;
    }
  }, []);

  const { events, connected } = useSSE(`${API_URL}/api/v1/flows/feed`, user ? handleIncomingEvent : undefined);

  // Fetch models to see their configurations
  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/v1/models`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setModels(data.map((m: any) => m.name));
        const active = data.find((m: any) => m.is_active);
        if (active) {
          setActiveModel(active.name);
          setThreshold(active.threshold);
        }
      }
    } catch { /* ignore */ }
  }, []);

  // Fetch KPIs
  const fetchKPIs = useCallback(async () => {
    setKpiLoading(true);
    setKpiError(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/stats/kpi`, { credentials: 'include' });
      if (res.ok) {
        setKpi(await res.json());
      } else {
        setKpiError('Failed to fetch KPI stats');
      }
    } catch {
      setKpiError('Connection error fetching KPIs');
    } finally {
      setKpiLoading(false);
    }
  }, []);

  // Fetch Timeline
  const fetchTimeline = useCallback(async () => {
    setTimelineLoading(true);
    setTimelineError(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/stats/timeline`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setTimeline(data.points || []);
      } else {
        setTimelineError('Failed to fetch timeline data');
      }
    } catch {
      setTimelineError('Connection error fetching timeline');
    } finally {
      setTimelineLoading(false);
    }
  }, []);

  // Fetch Alerts
  const fetchAlerts = useCallback(async () => {
    setAlertsLoading(true);
    setAlertsError(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/alerts?limit=100`, { credentials: 'include' });
      if (res.ok) {
        setAlerts(await res.json());
      } else {
        setAlertsError('Failed to fetch alerts list');
      }
    } catch {
      setAlertsError('Connection error fetching alerts');
    } finally {
      setAlertsLoading(false);
    }
  }, []);

  // Fetch Drift
  const fetchDrift = useCallback(async () => {
    setDriftLoading(true);
    setDriftError(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/drift`, { credentials: 'include' });
      if (res.ok) {
        setDrift(await res.json());
      } else {
        setDriftError('Failed to calculate traffic drift');
      }
    } catch {
      setDriftError('Connection error checking drift');
    } finally {
      setDriftLoading(false);
    }
  }, []);

  // Fetch Simulator state
  const fetchScenario = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/simulate`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setActiveScenario(data.active_scenario);
      }
    } catch { /* ignore */ }
  }, []);

  // Master fetch that triggers on timer intervals
  const fetchAllData = useCallback(() => {
    fetchKPIs();
    fetchTimeline();
    fetchAlerts();
    fetchDrift();
    fetchScenario();
  }, [fetchKPIs, fetchTimeline, fetchAlerts, fetchDrift, fetchScenario]);

  useEffect(() => {
    if (!user) return;

    fetchModels();
    fetchAllData();

    const fetchInterval = setInterval(fetchAllData, 5000);
    
    // Periodically update the live throughput rate displays
    const rateInterval = setInterval(() => {
      setFlowsPerSec(flowCounter.current / 2);
      setAlertRate(alertCounter.current / 2);
      flowCounter.current = 0;
      alertCounter.current = 0;
    }, 2000);

    return () => {
      clearInterval(fetchInterval);
      clearInterval(rateInterval);
    };
  }, [user, fetchModels, fetchAllData]);

  // Update threshold on the backend
  const handleThresholdChange = useCallback(async (value: number) => {
    setThreshold(value);
    try {
      await fetch(`${API_URL}/api/v1/models/${activeModel}/threshold`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: value }),
        credentials: 'include',
      });
    } catch { /* ignore */ }
  }, [activeModel]);

  // Update active model on the backend
  const handleActiveModelChange = useCallback(async (modelName: string) => {
    setActiveModel(modelName);
    try {
      const res = await fetch(`${API_URL}/api/v1/models/active`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName }),
        credentials: 'include',
      });
      if (res.ok) {
        fetchModels();
      }
    } catch { /* ignore */ }
  }, [fetchModels]);

  // Trigger scenario simulator on the backend
  const handleSimulate = useCallback(async (scenario: string | null) => {
    try {
      const res = await fetch(`${API_URL}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario }),
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        setActiveScenario(data.active_scenario);
      }
    } catch { /* ignore */ }
  }, []);

  // Submit analyst TP/FP verdict feedback
  const handleFeedbackSubmit = useCallback(async (alertId: string, verdict: string) => {
    try {
      const res = await fetch(`${API_URL}/api/v1/alerts/${alertId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verdict }),
        credentials: 'include',
      });
      if (res.ok) {
        // Update alerts list locally
        setAlerts(prev => prev.map(a => a.id === alertId ? { ...a, feedback_verdict: verdict } : a));
        if (selectedAlert && selectedAlert.id === alertId) {
          setSelectedAlert(prev => prev ? { ...prev, feedback_verdict: verdict } : null);
        }
      }
    } catch { /* ignore */ }
  }, [selectedAlert]);

  // Download analyst feedback CSV dataset
  const handleExportCSV = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/v1/alerts/feedback/export`, { credentials: 'include' });
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'analyst_feedback_dataset.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    } catch { /* ignore */ }
  }, []);

  // Logout current session
  const handleLogout = useCallback(async () => {
    try {
      await fetch(`${API_URL}/api/v1/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      });
      setUser(null);
    } catch { /* ignore */ }
  }, []);

  if (authLoading) {
    return (
      <div className="login-container">
        <div className="loading-spinner" style={{ width: 40, height: 40 }} />
        <div style={{ marginTop: 12 }}>Verifying session...</div>
      </div>
    );
  }

  if (!user) {
    return <LoginForm onLoginSuccess={(username) => setUser(username)} />;
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🛡️ Network Anomaly Detection</h1>
        <div className="header-controls">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Model:</span>
            <select
              className="model-select"
              value={activeModel}
              onChange={e => handleActiveModelChange(e.target.value)}
              id="model-selector"
            >
              {models.map(m => (
                <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
          <div className="threshold-control">
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Threshold:</span>
            <input
              type="range"
              className="threshold-slider"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={e => handleThresholdChange(parseFloat(e.target.value))}
              id="threshold-slider"
            />
            <span className="threshold-value">{threshold.toFixed(2)}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderLeft: '1px solid var(--border-subtle)', paddingLeft: '12px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>👤 {user}</span>
            <button className="close-btn" style={{ fontSize: '0.6875rem', padding: '2px 8px' }} onClick={handleLogout}>
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Top Status Strip */}
      <div className="status-strip">
        <div className="status-item">
          <span className="status-label">System Connection</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: connected ? 'var(--accent-green)' : 'var(--accent-red)',
              boxShadow: connected ? '0 0 6px var(--accent-green-glow)' : '0 0 6px var(--accent-red-glow)',
            }} />
            <span className="status-value">{connected ? 'Connected (Live)' : 'Disconnected'}</span>
          </div>
        </div>
        <div className="status-item">
          <span className="status-label">Active Classifier</span>
          <span className="status-value blue" style={{ marginTop: '4px' }}>
            {activeModel.replace(/_/g, ' ')} v1
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Throughput</span>
          <span className="status-value" style={{ fontVariantNumeric: 'tabular-nums', marginTop: '4px' }}>
            {flowsPerSec.toFixed(1)} flows/sec
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Alert Rate</span>
          <span className="status-value red" style={{ fontVariantNumeric: 'tabular-nums', marginTop: '4px' }}>
            {alertRate.toFixed(1)} alerts/sec
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Baseline Drift</span>
          {driftLoading ? (
            <span className="status-value" style={{ marginTop: '4px' }}>Calculating PSI...</span>
          ) : driftError ? (
            <span className="status-value red" style={{ marginTop: '4px' }}>Calculation error</span>
          ) : drift ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
              <span className={`badge ${drift.status === 'drifting' ? 'anomaly' : 'normal'}`}>
                {drift.status === 'drifting' ? 'Drifting' : 'Stable'} (PSI: {drift.overall_psi.toFixed(3)})
              </span>
              {drift.status === 'drifting' && (
                <span className="stale-notice">⚠ Stale Model</span>
              )}
            </div>
          ) : (
            <span className="status-value" style={{ marginTop: '4px' }}>No readings</span>
          )}
        </div>
      </div>

      <main className="main-content">
        <KPIStrip kpi={kpi} threshold={threshold} loading={kpiLoading} error={kpiError} />
        
        <TimelineChart data={timeline} threshold={threshold} loading={timelineLoading} error={timelineError} />
        
        <FlowFeed events={events} />
        
        <SystemOperations
          drift={drift}
          driftLoading={driftLoading}
          driftError={driftError}
          activeScenario={activeScenario}
          onSimulate={handleSimulate}
        />
        
        <AlertTable
          alerts={alerts}
          onSelect={setSelectedAlert}
          loading={alertsLoading}
          error={alertsError}
          onExportCSV={handleExportCSV}
        />
      </main>

      {selectedAlert && (
        <DrillDown
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onSubmitFeedback={handleFeedbackSubmit}
        />
      )}
    </div>
  );
}

export default App;
