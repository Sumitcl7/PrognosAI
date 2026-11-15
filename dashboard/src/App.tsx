import { useEffect, useState } from "react";

import {

  Activity,

  AlertTriangle,

  ArrowUpRight,

  Bell,

  ChevronDown,

  CircleGauge,

  Clock3,

  Cpu,

  LayoutDashboard,

  Menu,

  Search,

  Settings,

  ShieldCheck,

  Wrench,

  X,

  Zap,

} from "lucide-react";

import {

  Area,

  AreaChart,

  CartesianGrid,

  ReferenceArea,

  ReferenceLine,

  ResponsiveContainer,

  Tooltip,

  XAxis,

  YAxis,

} from "recharts";

import {

  getFleetEvaluation,

  getEngineHistory,

  type FleetEvaluation,

  type EnginePrediction,

  type EngineHistory,

} from "./api";

import "./App.css";

const navigation = [

  { label: "Overview", icon: LayoutDashboard },

  { label: "Fleet Monitoring", icon: Activity },

  { label: "RUL Predictions", icon: CircleGauge },

  { label: "Drift Monitoring", icon: Zap },

  { label: "Maintenance", icon: Wrench },

];

function formatEngineId(id: number) {

  return `ENG-${String(id).padStart(3, "0")}`;

}

function formatRisk(risk: string) {

  return risk.charAt(0) + risk.slice(1).toLowerCase();

}

function formatDrift(status: string) {

  switch (status) {

    case "stable":

      return "Stable";

    case "warning":

      return "Warning";

    case "drift_detected":

      return "Drift detected";

    case "not_evaluated":

      return "Not evaluated";

    default:

      return status;

  }

}

function getDriftClass(status: string) {

  switch (status) {

    case "stable":

      return "stable";

    case "warning":

      return "warning";

    case "drift_detected":

      return "review";

    default:

      return "not-evaluated";

  }

}

function App() {

  const [active, setActive] = useState("Overview");

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [fleet, setFleet] = useState<FleetEvaluation | null>(null);

  const [loading, setLoading] = useState(true);

  const [apiError, setApiError] = useState<string | null>(null);

  const [selectedEngineId, setSelectedEngineId] = useState(1);

  const [riskFilter, setRiskFilter] = useState("ALL");

  const [confidenceFilter, setConfidenceFilter] = useState("ALL");

  const [reviewFilter, setReviewFilter] = useState("ALL");

  const [intervalMode, setIntervalMode] = useState<"observation" | "engine">(

    "engine",

  );

  const [engineHistory, setEngineHistory] = useState<EngineHistory | null>(

    null,

  );

  const [historyLoading, setHistoryLoading] = useState(false);

  const [historyError, setHistoryError] = useState<string | null>(null);

  const [historyReload, setHistoryReload] = useState(0);

  // Load fleet evaluation.

  useEffect(() => {

    let cancelled = false;

    getFleetEvaluation()

      .then((data) => {

        if (!cancelled) {

          setFleet(data);

          setApiError(null);

        }

      })

      .catch((error: unknown) => {

        if (!cancelled) {

          setApiError(

            error instanceof Error

              ? error.message

              : "Unable to load fleet evaluation",

          );

        }

      })

      .finally(() => {

        if (!cancelled) {

          setLoading(false);

        }

      });

    return () => {

      cancelled = true;

    };

  }, []);

  // Load selected engine history.

  useEffect(() => {

    let cancelled = false;

    setHistoryLoading(true);

    setHistoryError(null);

    setEngineHistory(null);

    getEngineHistory(selectedEngineId)

      .then((data) => {

        if (!cancelled) {

          setEngineHistory(data);

        }

      })

      .catch((error: unknown) => {

        if (!cancelled) {

          setHistoryError(

            error instanceof Error

              ? error.message

              : "Unable to load engine history",

          );

        }

      })

      .finally(() => {

        if (!cancelled) {

          setHistoryLoading(false);

        }

      });

    return () => {

      cancelled = true;

    };

  }, [selectedEngineId, historyReload]);

  const summary = fleet?.summary;

  const engines = fleet?.engines ?? [];

  const filteredEngines = engines.filter((engine) => {

    const matchesRisk =

      riskFilter === "ALL" || engine.risk_level === riskFilter;

    const matchesConfidence =

      confidenceFilter === "ALL" ||

      engine.confidence_category === confidenceFilter;

    const matchesReview =

      reviewFilter === "ALL" ||

      (reviewFilter === "REQUIRED" && engine.model_review_required) ||

      (reviewFilter === "NOT_REQUIRED" && !engine.model_review_required);

    return matchesRisk && matchesConfidence && matchesReview;

  });

  const riskCounts = summary?.risk_counts ?? {};

  const driftCounts = summary?.drift_counts ?? {};

  const totalEngines = fleet?.evaluation.total_engines ?? 0;

  const criticalEngines = riskCounts.CRITICAL ?? 0;

  const maintenanceAlerts = summary?.maintenance_alerts ?? 0;

  const averageRul = summary?.average_predicted_rul ?? 0;

  const selectedEngine = engines.find(

    (engine) => engine.engine_id === selectedEngineId,

  );

  const latestPrediction =

    engineHistory && engineHistory.predictions.length > 0

      ? engineHistory.predictions[engineHistory.predictions.length - 1]

      : undefined;

  const selectedIntervalWidth = latestPrediction

    ? intervalMode === "engine"

      ? latestPrediction.engine_interval.width

      : latestPrediction.observation_interval.width

    : null;

  // Review information from the actual fleet API.

  const reviewReasons: string[] = [];

  if (selectedEngine?.confidence_category === "LOW") {

    reviewReasons.push("Wide prediction uncertainty");

  }

  if (selectedEngine?.drift_status === "warning") {

    reviewReasons.push("Sensor drift warning");

  }

  if (selectedEngine?.drift_status === "drift_detected") {

    reviewReasons.push("Sensor drift detected");

  }

  if (selectedEngine?.drift_status === "not_evaluated") {

    reviewReasons.push("Drift monitoring not evaluated");

  }

  const confidenceReviewRequired =

    selectedEngine?.model_review_required ?? false;

  // Determine the current risk zone from the latest RUL prediction.

  const currentRul = latestPrediction?.predicted_rul;

  const riskZone =

    currentRul === undefined

      ? null

      : currentRul <= 15

        ? {

            label: "CRITICAL RISK",

            action: "URGENT MAINTENANCE REVIEW",

            y1: 0,

            y2: 15,

            color: "#ef6b6b",

          }

        : currentRul <= 30

          ? {

              label: "HIGH RISK",

              action: "SCHEDULE MAINTENANCE SOON",

              y1: 15,

              y2: 30,

              color: "#f0ad4e",

            }

          : currentRul <= 60

            ? {

                label: "MODERATE RISK",

                action: "PLAN MAINTENANCE",

                y1: 30,

                y2: 60,

                color: "#36a5e8",

              }

            : {

                label: "LOW RISK",

                action: "CONTINUE MONITORING",

                y1: 60,

                y2: 125,

                color: "#35c99a",

              };

  // Select the appropriate uncertainty interval for every cycle.

  const chartData = (engineHistory?.predictions ?? []).map((point) => {

    const interval =

      intervalMode === "engine"

        ? point.engine_interval

        : point.observation_interval;

    return {

      cycle: point.cycle,

      rul: point.predicted_rul,

      lower: interval.lower_bound,

      upper: interval.upper_bound,

      band: Math.max(0, interval.upper_bound - interval.lower_bound),

    };

  });

  const latestInterval =

    intervalMode === "engine"

      ? latestPrediction?.engine_interval

      : latestPrediction?.observation_interval;

  const priorityEngines = engines

    .filter(

      (engine) =>

        engine.risk_level === "CRITICAL" ||

        engine.risk_level === "HIGH" ||

        engine.model_review_required,

    )

    .sort((a, b) => {

      const riskOrder: Record<string, number> = {

        CRITICAL: 0,

        HIGH: 1,

        MODERATE: 2,

        LOW: 3,

      };

      return (riskOrder[a.risk_level] ?? 4) - (riskOrder[b.risk_level] ?? 4);

    })

    .slice(0, 4);

  return (

    <div className="app-shell">

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>

        <div className="brand">

          <div className="brand-icon">

            <Activity size={23} />

          </div>

          <div>

            <h2>

              PROGNOS<span>AI</span>

            </h2>

            <p>INTELLIGENCE PLATFORM</p>

          </div>

          <button

            className="mobile-close"

            onClick={() => setSidebarOpen(false)}

            aria-label="Close menu"

          >

            <X />

          </button>

        </div>

        <div className="workspace-label">WORKSPACE</div>

        <div className="workspace">

          <div className="workspace-icon">

            <Cpu size={19} />

          </div>

          <div>

            <strong>Industrial Fleet</strong>

            <small>FD001 evaluation cohort</small>

          </div>

          <ChevronDown size={15} />

        </div>

        <div className="nav-label">MONITORING</div>

        <nav>

          {navigation.map((item) => {

            const Icon = item.icon;

            return (

              <button

                key={item.label}

                className={`nav-item ${active === item.label ? "active" : ""}`}

                onClick={() => {

                  setActive(item.label);

                  setSidebarOpen(false);

                }}

              >

                <Icon size={18} />

                <span>{item.label}</span>

                {item.label === "Maintenance" && maintenanceAlerts > 0 && (

                  <span className="nav-count">{maintenanceAlerts}</span>

                )}

              </button>

            );

          })}

        </nav>

        <div className="sidebar-bottom">

          <div className="system-status">

            <span className="status-dot" />

            <div>

              <strong>

                {apiError ? "API unavailable" : "Evaluation service"}

              </strong>

              <small>

                {loading

                  ? "Connecting..."

                  : apiError

                    ? "Connection error"

                    : "Response received"}

              </small>

            </div>

          </div>

          <button className="nav-item settings">

            <Settings size={18} />

            Settings

          </button>

          <div className="profile">

            <div className="avatar">SK</div>

            <div>

              <strong>Administrator</strong>

              <small>Fleet operator</small>

            </div>

            <ChevronDown size={15} />

          </div>

        </div>

      </aside>

      {sidebarOpen && (

        <div

          className="sidebar-overlay"

          onClick={() => setSidebarOpen(false)}

        />

      )}

      <main className="main-area">

        <header className="topbar">

          <div className="breadcrumb">

            <button

              className="mobile-menu"

              onClick={() => setSidebarOpen(true)}

              aria-label="Open menu"

            >

              <Menu />

            </button>

            <span>Workspace</span>

            <span className="breadcrumb-separator">/</span>

            <strong>{active}</strong>

          </div>

          <div className="topbar-actions">

            <div className="live-indicator">

              <span className="status-dot" /> OFFLINE EVALUATION

            </div>

            <button className="icon-button" aria-label="Search">

              <Search size={19} />

            </button>

            <button

              className="icon-button notification"

              aria-label="Notifications"

            >

              <Bell size={19} />

              <i />

            </button>

            <div className="topbar-date">FD001 DATASET</div>

          </div>

        </header>

        <div className="dashboard-content">

          <section className="page-heading">

            <div>

              <div className="eyebrow">

                PREDICTIVE INTELLIGENCE <span>•</span> NASA C-MAPSS FD001

              </div>

              <h1>Fleet Overview</h1>

              <p>

                Remaining useful life predictions, uncertainty and maintenance

                decision support.

              </p>

            </div>

            <div className="date-button">

              <Clock3 size={16} />

              Offline evaluation

              <ChevronDown size={15} />

            </div>

          </section>

          {apiError && (

            <div className="panel" style={{ padding: 18, marginBottom: 20 }}>

              <strong>Unable to load fleet data</strong>

              <p>{apiError}</p>

              <p>

                Check that the FastAPI server is running on port 8000 and the

                Vite proxy is configured.

              </p>

              <button

                className="subtle-button"

                onClick={() => window.location.reload()}

              >

                Retry <ArrowUpRight size={14} />

              </button>

            </div>

          )}

          {loading && (

            <div className="panel" style={{ padding: 22, marginBottom: 20 }}>

              Loading fleet evaluation from PrognosAI API...

            </div>

          )}

          <section className="metrics-grid">

            <MetricCard

              title="Total Engines"

              value={loading ? "—" : String(totalEngines)}

              detail="Evaluation cohort"

              icon={<Cpu />}

              tone="blue"

              trend="NASA FD001"

            />

            <MetricCard

              title="Critical Engines"

              value={loading ? "—" : String(criticalEngines)}

              detail="Urgent review category"

              icon={<AlertTriangle />}

              tone="red"

              trend={`${

                totalEngines

                  ? ((criticalEngines / totalEngines) * 100).toFixed(1)

                  : "0.0"

              }% of cohort`}

            />

            <MetricCard

              title="Maintenance Alerts"

              value={loading ? "—" : String(maintenanceAlerts)}

              detail="High and critical risk"

              icon={<Wrench />}

              tone="orange"

              trend="Priority queue"

            />

            <MetricCard

              title="Average Predicted RUL"

              value={loading ? "—" : averageRul.toFixed(1)}

              detail="Across evaluation cohort"

              icon={<ShieldCheck />}

              tone="green"

              trend="Cycles"

            />

          </section>

          <section className="main-grid">

            <div className="panel prediction-panel">

              <div className="panel-header">

                <div>

                  <div className="panel-kicker">DEGRADATION ANALYTICS</div>

                  <h3>Remaining Useful Life</h3>

                  <p>Actual model predictions across operating cycles</p>

                </div>

                <select

                  className="subtle-button"

                  value={selectedEngineId}

                  onChange={(event) =>

                    setSelectedEngineId(Number(event.target.value))

                  }

                  aria-label="Select engine"

                  disabled={historyLoading}

                >

                  {Array.from({ length: 100 }, (_, index) => index + 1).map(

                    (id) => (

                      <option key={id} value={id}>

                        {formatEngineId(id)}

                      </option>

                    ),

                  )}

                </select>

              </div>

              <div

                style={{

                  display: "flex",

                  justifyContent: "space-between",

                  alignItems: "center",

                  flexWrap: "wrap",

                  gap: 12,

                  marginBottom: 16,

                }}

              >

                <div className="chart-legend">

                  <span>

                    <i className="legend-line" /> Predicted RUL

                  </span>

                  <span>

                    <i className="legend-area" /> Uncertainty interval

                  </span>

                </div>

                <div style={{ display: "flex", gap: 8 }}>

                  <button

                    className="subtle-button"

                    onClick={() => setIntervalMode("observation")}

                    style={{

                      opacity: intervalMode === "observation" ? 1 : 0.55,

                      border:

                        intervalMode === "observation"

                          ? "1px solid #36a5e8"

                          : undefined,

                    }}

                  >

                    Observation

                  </button>

                  <button

                    className="subtle-button"

                    onClick={() => setIntervalMode("engine")}

                    style={{

                      opacity: intervalMode === "engine" ? 1 : 0.55,

                      border:

                        intervalMode === "engine"

                          ? "1px solid #36a5e8"

                          : undefined,

                    }}

                  >

                    Engine-level

                  </button>

                </div>

              </div>

              {riskZone && (

                <div

                  style={{

                    display: "flex",

                    alignItems: "center",

                    justifyContent: "space-between",

                    gap: 12,

                    flexWrap: "wrap",

                    padding: "11px 14px",

                    marginTop: 12,

                    marginBottom: 12,

                    borderRadius: 8,

                    background: `${riskZone.color}14`,

                    border: `1px solid ${riskZone.color}55`,

                  }}

                >

                  <div

                    style={{

                      display: "flex",

                      alignItems: "center",

                      gap: 9,

                    }}

                  >

                    <span

                      style={{

                        width: 9,

                        height: 9,

                        borderRadius: "50%",

                        background: riskZone.color,

                      }}

                    />

                    <strong

                      style={{

                        color: riskZone.color,

                        fontSize: 12,

                      }}

                    >

                      {riskZone.label}

                    </strong>

                  </div>

                  <span style={{ fontSize: 12, color: "#a9bad0" }}>

                    {riskZone.action}

                  </span>

                </div>

              )}

              {historyLoading && (

                <div

                  className="chart-container"

                  style={{

                    display: "flex",

                    alignItems: "center",

                    justifyContent: "center",

                  }}

                >

                  <p>Loading {formatEngineId(selectedEngineId)} history...</p>

                </div>

              )}

              {!historyLoading && historyError && (

                <div

                  className="chart-container"

                  style={{

                    display: "flex",

                    alignItems: "center",

                    justifyContent: "center",

                    flexDirection: "column",

                    gap: 12,

                  }}

                >

                  <AlertTriangle size={28} />

                  <strong>Unable to load engine history</strong>

                  <p>{historyError}</p>

                  <button

                    className="subtle-button"

                    onClick={() => setHistoryReload((value) => value + 1)}

                  >

                    Retry

                  </button>

                </div>

              )}

              {!historyLoading && !historyError && chartData.length > 0 && (

                <div className="chart-container">

                  <ResponsiveContainer width="100%" height="100%">

                    <AreaChart

                      data={chartData}

                      margin={{

                        top: 10,

                        right: 8,

                        left: -18,

                        bottom: 0,

                      }}

                    >

                      <defs>

                        <linearGradient

                          id="rulGradient"

                          x1="0"

                          y1="0"

                          x2="0"

                          y2="1"

                        >

                          <stop

                            offset="0%"

                            stopColor="#36a5e8"

                            stopOpacity={0.25}

                          />

                          <stop

                            offset="100%"

                            stopColor="#36a5e8"

                            stopOpacity={0.01}

                          />

                        </linearGradient>

                      </defs>

                      <CartesianGrid

                        stroke="#202d40"

                        strokeDasharray="3 5"

                        vertical={false}

                      />

                      {riskZone && (

                        <ReferenceArea

                          y1={riskZone.y1}

                          y2={riskZone.y2}

                          fill={riskZone.color}

                          fillOpacity={0.12}

                          stroke={riskZone.color}

                          strokeOpacity={0.35}

                          strokeDasharray="4 4"

                        />

                      )}

                      <XAxis

                        type="number"

                        dataKey="cycle"

                        domain={["dataMin", "dataMax"]}

                        stroke="#687990"

                        tickLine={false}

                        axisLine={false}

                        tick={{ fontSize: 11 }}

                        tickCount={7}

                        label={{

                          value: "Operating cycles",

                          position: "insideBottom",

                          offset: -2,

                          fill: "#687990",

                          fontSize: 10,

                        }}

                      />

                      <YAxis

                        type="number"

                        domain={[0, 125]}

                        stroke="#687990"

                        tickLine={false}

                        axisLine={false}

                        tick={{ fontSize: 11 }}

                        label={{

                          value: "RUL (cycles)",

                          angle: -90,

                          position: "insideLeft",

                          fill: "#687990",

                          fontSize: 10,

                        }}

                      />

                      <Tooltip

                        labelFormatter={(value) => `Cycle ${value}`}

                        formatter={(value, name) => [

                          Number(value).toFixed(1),

                          name === "rul"

                            ? "Predicted RUL"

                            : name === "upper"

                              ? "Upper bound"

                              : name === "lower"

                                ? "Lower bound"

                                : String(name),

                        ]}

                        contentStyle={{

                          background: "#101a2a",

                          border: "1px solid #263750",

                          borderRadius: 8,

                          color: "#fff",

                        }}

                      />

                      <Area

                        type="monotone"

                        dataKey="lower"

                        stackId="interval"

                        stroke="none"

                        fill="#36a5e8"

                        fillOpacity={0}

                        legendType="none"

                        tooltipType="none"

                      />

                      <Area

                        type="monotone"

                        dataKey="band"

                        stackId="interval"

                        stroke="none"

                        fill="#36a5e8"

                        fillOpacity={0.15}

                        name="Uncertainty interval"

                      />

                      <Area

                        type="monotone"

                        dataKey="rul"

                        stroke="#36a5e8"

                        strokeWidth={2.5}

                        fill="url(#rulGradient)"

                        activeDot={{

                          r: 5,

                          fill: "#36a5e8",

                        }}

                        name="Predicted RUL"

                      />

                      <ReferenceLine

                        y={60}

                        stroke="#36a5e8"

                        strokeDasharray="5 5"

                        strokeWidth={1.5}

                        label={{

                          value: "PLAN MAINTENANCE · 60",

                          position: "insideTopRight",

                          fill: "#36a5e8",

                          fontSize: 10,

                        }}

                      />

                      <ReferenceLine

                        y={30}

                        stroke="#f0ad4e"

                        strokeDasharray="5 5"

                        strokeWidth={1.5}

                        label={{

                          value: "SCHEDULE SOON · 30",

                          position: "insideTopRight",

                          fill: "#f0ad4e",

                          fontSize: 10,

                        }}

                      />

                      <ReferenceLine

                        y={15}

                        stroke="#ef6b6b"

                        strokeDasharray="5 5"

                        strokeWidth={1.5}

                        label={{

                          value: "URGENT · 15",

                          position: "insideTopRight",

                          fill: "#ef6b6b",

                          fontSize: 10,

                        }}

                      />

                    </AreaChart>

                  </ResponsiveContainer>

                </div>

              )}

              {!historyLoading && !historyError && chartData.length === 0 && (

                <div className="chart-container">

                  No prediction history returned for this engine.

                </div>

              )}

              <div className="chart-footer">

                <span>

                  {engineHistory

                    ? `${engineHistory.history_observations} OBSERVATIONS`

                    : "OPERATING CYCLES"}

                </span>

                <span>

                  {intervalMode === "engine"

                    ? "ENGINE-LEVEL INTERVAL"

                    : "OBSERVATION-LEVEL INTERVAL"}

                </span>

              </div>

              {latestPrediction && latestInterval && (

                <div className="health-bars" style={{ marginTop: 18 }}>

                  <div className="health-label">

                    <span>Selected engine</span>

                    <strong>{formatEngineId(selectedEngineId)}</strong>

                  </div>

                  <div className="health-label">

                    <span>Latest operating cycle</span>

                    <strong>{latestPrediction.cycle}</strong>

                  </div>

                  <div className="health-label">

                    <span>Predicted RUL</span>

                    <strong>

                      {latestPrediction.predicted_rul.toFixed(1)} cycles

                    </strong>

                  </div>

                  <div className="health-label">

                    <span>Predicted stage</span>

                    <strong>{latestPrediction.predicted_stage}</strong>

                  </div>

                  <div className="health-label">

                    <span>Lower / upper bound</span>

                    <strong>

                      {latestInterval.lower_bound.toFixed(1)} –{" "}

                      {latestInterval.upper_bound.toFixed(1)}

                    </strong>

                  </div>

                  <div className="health-label">

                    <span>Interval width</span>

                    <strong>{latestInterval.width.toFixed(1)} cycles</strong>

                  </div>

                </div>

              )}

              {selectedEngine && (

                <div

                  style={{

                    marginTop: 16,

                    paddingTop: 14,

                    borderTop: "1px solid #202d40",

                    display: "flex",

                    justifyContent: "space-between",

                    flexWrap: "wrap",

                    gap: 12,

                  }}

                >

                  <span>

                    Risk:{" "}

                    <strong>{formatRisk(selectedEngine.risk_level)}</strong>

                  </span>

                  <span>

                    Maintenance:{" "}

                    <strong>

                      {selectedEngine.maintenance_action.replaceAll("_", " ")}

                    </strong>

                  </span>

                  <span>

                    Drift:{" "}

                    <strong>{formatDrift(selectedEngine.drift_status)}</strong>

                  </span>

                </div>

              )}

              {selectedEngine && (

                <div

                  style={{

                    marginTop: 18,

                    padding: 16,

                    borderRadius: 10,

                    background: "#111d2d",

                    border: "1px solid #263750",

                  }}

                >

                  <div className="panel-kicker">MODEL CONFIDENCE & REVIEW</div>

                  <div

                    style={{

                      display: "grid",

                      gridTemplateColumns:

                        "repeat(auto-fit, minmax(150px, 1fr))",

                      gap: 14,

                      marginTop: 14,

                    }}

                  >

                    <div>

                      <span style={{ color: "#8da4bf", fontSize: 11 }}>

                        PREDICTION CONFIDENCE

                      </span>

                      <div

                        style={{

                          marginTop: 7,

                          fontWeight: 700,

                          color:

                            selectedEngine.confidence_category === "HIGH"

                              ? "#35c99a"

                              : selectedEngine.confidence_category === "MEDIUM"

                                ? "#36a5e8"

                                : "#f0ad4e",

                        }}

                      >

                        {selectedEngine.confidence_category}

                      </div>

                    </div>

                    <div>

                      <span style={{ color: "#8da4bf", fontSize: 11 }}>

                        INTERVAL WIDTH

                      </span>

                      <div style={{ marginTop: 7, fontWeight: 700 }}>

                        {selectedIntervalWidth?.toFixed(1)} cycles

                      </div>

                    </div>

                    <div>

                      <span style={{ color: "#8da4bf", fontSize: 11 }}>

                        MODEL REVIEW

                      </span>

                      <div

                        style={{

                          marginTop: 7,

                          fontWeight: 700,

                          color: confidenceReviewRequired

                            ? "#f0ad4e"

                            : "#35c99a",

                        }}

                      >

                        {confidenceReviewRequired ? "REQUIRED" : "NOT REQUIRED"}

                      </div>

                    </div>

                  </div>

                  <div

                    style={{

                      marginTop: 16,

                      paddingTop: 13,

                      borderTop: "1px solid #263750",

                    }}

                  >

                    <div

                      style={{

                        color: "#8da4bf",

                        fontSize: 11,

                        marginBottom: 8,

                      }}

                    >

                      REVIEW REASONS / MONITORING NOTES

                    </div>

                    {reviewReasons.length > 0 ? (

                      <div

                        style={{

                          display: "flex",

                          flexWrap: "wrap",

                          gap: 8,

                        }}

                      >

                        {reviewReasons.map((reason) => (

                          <span

                            key={reason}

                            style={{

                              padding: "6px 10px",

                              borderRadius: 6,

                              background: "#263344",

                              color: "#e5b96d",

                              fontSize: 11,

                            }}

                          >

                            {reason}

                          </span>

                        ))}

                      </div>

                    ) : (

                      <span style={{ color: "#35c99a", fontSize: 12 }}>

                        No additional monitoring notes

                      </span>

                    )}

                  </div>

                </div>

              )}

              <p className="disclaimer" style={{ marginTop: 14 }}>

                Predictions are generated from the frozen FD001 model.

                Uncertainty intervals represent calibrated prediction ranges.

                The official test labels are not used as prediction inputs.

              </p>

            </div>

            <div className="panel health-panel">

              <div className="panel-header">

                <div>

                  <div className="panel-kicker">FLEET CONDITION</div>

                  <h3>Engine Risk Distribution</h3>

                  <p>Actual model risk categories</p>

                </div>

              </div>

              <div className="health-total">

                <strong>{loading ? "—" : totalEngines}</strong>

                <span>

                  ENGINES

                  <br />

                  ANALYZED

                </span>

              </div>

              <div className="health-bars">

                <HealthBar

                  label="Low risk"

                  count={riskCounts.LOW ?? 0}

                  percent={

                    totalEngines

                      ? ((riskCounts.LOW ?? 0) / totalEngines) * 100

                      : 0

                  }

                  color="green"

                />

                <HealthBar

                  label="Moderate"

                  count={riskCounts.MODERATE ?? 0}

                  percent={

                    totalEngines

                      ? ((riskCounts.MODERATE ?? 0) / totalEngines) * 100

                      : 0

                  }

                  color="blue"

                />

                <HealthBar

                  label="High risk"

                  count={riskCounts.HIGH ?? 0}

                  percent={

                    totalEngines

                      ? ((riskCounts.HIGH ?? 0) / totalEngines) * 100

                      : 0

                  }

                  color="orange"

                />

                <HealthBar

                  label="Critical"

                  count={riskCounts.CRITICAL ?? 0}

                  percent={

                    totalEngines

                      ? ((riskCounts.CRITICAL ?? 0) / totalEngines) * 100

                      : 0

                  }

                  color="red"

                />

              </div>

              <div className="health-note">

                <ShieldCheck size={16} />

                Based on model-predicted RUL

              </div>

            </div>

          </section>

          <section className="panel engine-panel">
            <div className="panel-header engine-header">
              <div>
                <div className="panel-kicker">OFFLINE EVALUATION</div>
                <h3>Engine Monitoring</h3>
                <p>Individual engine predictions and maintenance status</p>
              </div>
              <span className="subtle-button">
                {filteredEngines.length} of {engines.length} engines
              </span>
            </div>

            <div
              style={{
                display: "flex",
                alignItems: "end",
                flexWrap: "wrap",
                gap: 12,
                padding: "4px 0 18px",
              }}
            >
              <label style={{ display: "grid", gap: 6, color: "#8da4bf", fontSize: 11 }}>
                RISK LEVEL
                <select
                  className="subtle-button"
                  value={riskFilter}
                  onChange={(event) => setRiskFilter(event.target.value)}
                >
                  <option value="ALL">All risk levels</option>
                  <option value="CRITICAL">Critical</option>
                  <option value="HIGH">High</option>
                  <option value="MODERATE">Moderate</option>
                  <option value="LOW">Low</option>
                </select>
              </label>

              <label style={{ display: "grid", gap: 6, color: "#8da4bf", fontSize: 11 }}>
                CONFIDENCE
                <select
                  className="subtle-button"
                  value={confidenceFilter}
                  onChange={(event) => setConfidenceFilter(event.target.value)}
                >
                  <option value="ALL">All confidence</option>
                  <option value="HIGH">High</option>
                  <option value="MEDIUM">Medium</option>
                  <option value="LOW">Low</option>
                </select>
              </label>

              <label style={{ display: "grid", gap: 6, color: "#8da4bf", fontSize: 11 }}>
                MODEL REVIEW
                <select
                  className="subtle-button"
                  value={reviewFilter}
                  onChange={(event) => setReviewFilter(event.target.value)}
                >
                  <option value="ALL">All engines</option>
                  <option value="REQUIRED">Review required</option>
                  <option value="NOT_REQUIRED">No review required</option>
                </select>
              </label>

              <button
                className="subtle-button"
                onClick={() => {
                  setRiskFilter("ALL");
                  setConfidenceFilter("ALL");
                  setReviewFilter("ALL");
                }}
                disabled={
                  riskFilter === "ALL" &&
                  confidenceFilter === "ALL" &&
                  reviewFilter === "ALL"
                }
              >
                Reset filters
              </button>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ENGINE ID</th>
                    <th>REMAINING USEFUL LIFE</th>
                    <th>RISK LEVEL</th>
                    <th>CONFIDENCE</th>
                    <th>DRIFT STATUS</th>
                    <th>ACTION</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEngines.map((engine) => (
                    <EngineRow
                      key={engine.engine_id}
                      engine={engine}
                      onSelect={() => setSelectedEngineId(engine.engine_id)}
                      selected={selectedEngineId === engine.engine_id}
                    />
                  ))}

                  {!loading && engines.length === 0 && (
                    <tr>
                      <td colSpan={6}>
                        No engine records returned by the API.
                      </td>
                    </tr>
                  )}

                  {!loading && engines.length > 0 && filteredEngines.length === 0 && (
                    <tr>
                      <td colSpan={6}>
                        No engines match these filters. Try resetting the filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="table-footer">
              <span>
                Showing {filteredEngines.length} matching engines of {totalEngines}
              </span>
              <span>Official test cohort · Select a row to view its history</span>
            </div>
          </section>
          <section className="bottom-grid">

            <div className="panel alert-panel">

              <div className="panel-header">

                <div>

                  <div className="panel-kicker">ATTENTION REQUIRED</div>

                  <h3>Maintenance Alerts</h3>

                </div>

                <span className="alert-count">

                  {priorityEngines.length} PRIORITY

                </span>

              </div>

              {priorityEngines.length > 0 ? (

                priorityEngines.map((engine) => (

                  <AlertItem

                    key={engine.engine_id}

                    id={formatEngineId(engine.engine_id)}

                    message={engine.maintenance_action.replaceAll("_", " ")}

                    time={

                      engine.risk_level === "CRITICAL"

                        ? "Critical risk"

                        : engine.risk_level === "HIGH"

                          ? "High risk"

                          : engine.model_review_required

                            ? "Model review required"

                            : "Maintenance review"

                    }

                    tone={

                      engine.risk_level === "CRITICAL"

                        ? "red"

                        : engine.risk_level === "HIGH"

                          ? "orange"

                          : "blue"

                    }

                  />

                ))

              ) : (

                <p style={{ padding: 16 }}>

                  {loading

                    ? "Loading alerts..."

                    : "No priority alerts returned."}

                </p>

              )}

              <p className="disclaimer">

                Alerts are generated from model risk and review decisions.

              </p>

            </div>

            <div className="panel drift-panel">

              <div className="panel-header">

                <div>

                  <div className="panel-kicker">MODEL RELIABILITY</div>

                  <h3>Drift Monitoring</h3>

                  <p>Calibrated sensor monitoring</p>

                </div>

                <Activity size={20} className="drift-icon" />

              </div>

              <div className="drift-summary">

                <div>

                  <span className="status-dot" />

                  <strong>{driftCounts.stable ?? 0} stable engines</strong>

                </div>

                <span>Offline evaluation</span>

              </div>

              <div className="drift-stats">

                <div>

                  <span>Not evaluated</span>

                  <strong>{driftCounts.not_evaluated ?? 0}</strong>

                </div>

                <div>

                  <span>Drift warnings / detected</span>

                  <strong>

                    {(driftCounts.warning ?? 0) +

                      (driftCounts.drift_detected ?? 0)}

                  </strong>

                </div>

              </div>

              <p className="disclaimer">

                Drift is evaluated only when an eligible 20-cycle,

                stage-homogeneous window is available. Not evaluated does not

                mean stable.

              </p>

            </div>

          </section>

          <footer className="footer">

            <span>

              PROGNOSAI <i>•</i> INDUSTRIAL PREDICTIVE MAINTENANCE

            </span>

            <span>

              <span className="status-dot" />{" "}

              {apiError

                ? "API disconnected"

                : loading

                  ? "Connecting to API"

                  : "API connected"}{" "}

              <i>•</i> NASA C-MAPSS FD001

            </span>

          </footer>

        </div>

      </main>

    </div>

  );

}

function EngineRow({

  engine,

  onSelect,

  selected,

}: {

  engine: EnginePrediction;

  onSelect: () => void;

  selected: boolean;

}) {

  const status = formatRisk(engine.risk_level);

  const confidence = engine.confidence_category.toLowerCase();

  const driftClass = getDriftClass(engine.drift_status);

  return (

    <tr

      onClick={onSelect}

      style={{

        cursor: "pointer",

        background: selected ? "rgba(54, 165, 232, 0.07)" : undefined,

      }}

      title={`View ${formatEngineId(engine.engine_id)} history`}

    >

      <td className="engine-id">

        {formatEngineId(engine.engine_id)}

        {selected ? " •" : ""}

      </td>

      <td>

        <strong className="rul-value">

          {engine.predicted_rul.toFixed(1)} <small>cycles</small>

        </strong>

      </td>

      <td>

        <span className={`pill ${status.toLowerCase()}`}>{status}</span>

      </td>

      <td>

        <span className={`confidence ${confidence}`}>

          {engine.confidence_category}

        </span>

      </td>

      <td>

        <span className={`drift ${driftClass}`}>

          <i />

          {formatDrift(engine.drift_status)}

        </span>

      </td>

      <td>

        <span className="row-action" title={engine.maintenance_action}>

          <ArrowUpRight size={16} />

        </span>

      </td>

    </tr>

  );

}

function MetricCard({

  title,

  value,

  detail,

  icon,

  tone,

  trend,

}: {

  title: string;

  value: string;

  detail: string;

  icon: React.ReactNode;

  tone: string;

  trend: string;

}) {

  return (

    <div className="metric-card">

      <div className="metric-top">

        <span>{title}</span>

        <div className={`metric-icon ${tone}`}>{icon}</div>

      </div>

      <div className="metric-value">{value}</div>

      <div className="metric-bottom">

        <span>{detail}</span>

        <span>{trend}</span>

      </div>

    </div>

  );

}

function HealthBar({

  label,

  count,

  percent,

  color,

}: {

  label: string;

  count: number;

  percent: number;

  color: string;

}) {

  return (

    <div className="health-row">

      <div className="health-label">

        <span>

          <i className={color} />

          {label}

        </span>

        <strong>{count}</strong>

      </div>

      <div className="bar-track">

        <div className={`bar-fill ${color}`} style={{ width: `${percent}%` }} />

      </div>

    </div>

  );

}

function AlertItem({

  id,

  message,

  time,

  tone,

}: {

  id: string;

  message: string;

  time: string;

  tone: string;

}) {

  return (

    <div className="alert-item">

      <div className={`alert-symbol ${tone}`}>

        <AlertTriangle size={16} />

      </div>

      <div className="alert-message">

        <strong>{id}</strong>

        <span>{message}</span>

      </div>

      <span className="alert-time">{time}</span>

    </div>

  );

}

export default App;
