import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import "./App.css";

const money = (v) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(v || 0);

function StatusBadge({ label }) {
  const normalized = String(label || "Unknown");
  const tone =
    normalized === "Healthy"
      ? "ok"
      : normalized === "Processing"
      ? "processing"
      : ["Failed", "Warning", "Delayed"].includes(normalized)
      ? "warn"
      : "neutral";
  return (
    <span className={`badge ${tone}`}>
      {normalized}
    </span>
  );
}

function App() {
  const [dashboard, setDashboard] = useState();
  const [replay, setReplay] = useState();
  const [diagnostics, setDiagnostics] = useState();
  const [replayOptions, setReplayOptions] = useState();
  const [mlStatus, setMlStatus] = useState();
  const [analysis, setAnalysis] = useState();
  const [upload, setUpload] = useState();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const [batchSize, setBatchSize] = useState(100);
  const [intervalSeconds, setIntervalSeconds] = useState(5);
  const [batchMode, setBatchMode] = useState("events");
  const [batchValue, setBatchValue] = useState("");

  const refresh = async () => {
    try {
      const [d, r, x, options, ml] = await Promise.all([
        api.dashboard(),
        api.replay(),
        api.diagnostics(),
        api.replayOptions(),
        api.mlStatus(),
      ]);
      setDashboard(d);
      setReplay(r);
      setDiagnostics(x);
      setReplayOptions(options);
      setMlStatus(ml);
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, []);

  const progressPct = useMemo(() => {
    const total = replay?.total_events || 0;
    if (!total) return 0;
    return Math.min(100, ((replay?.events_published || 0) / total) * 100);
  }, [replay]);

  const startReplay = async () => {
    try {
      await api.startReplay({
        batch_size: Number(batchSize),
        interval_seconds: Number(intervalSeconds),
        batch_mode: batchMode,
        batch_value: batchMode === "events" ? null : batchValue,
      });
      await refresh();
    } catch (e) {
      setError(e.message);
    }
  };

  const controlReplay = async (action) => {
    try {
      await api.controlReplay(action);
      await refresh();
    } catch (e) {
      setError(e.message);
    }
  };

  const runAnalysis = async (task) => {
    try {
      setAnalysis(await api.analytics(task));
      setError("");
    } catch (e) {
      setError(e.message);
    }
  };

  const preview = async (file) => {
    if (!file) return;
    try {
      setUpload(await api.previewUpload(file));
      setError("");
    } catch (e) {
      setError(e.message);
    }
  };

  const modeOptions =
    batchMode === "day"
      ? replayOptions?.days || []
      : batchMode === "week"
      ? replayOptions?.weeks || []
      : batchMode === "store"
      ? replayOptions?.stores || []
      : [];

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">Retail operations dashboard</p>
          <h1>Store performance, as it happens.</h1>
          <p className="sub">Track real replay processing, operational health, and business results.</p>
        </div>
        <button className="secondary" onClick={refresh}>Refresh data</button>
      </header>

      {error && <p className="error">{error}</p>}
      {loading && <p className="sub">Loading data...</p>}

      <section className="kpis">
        {[
          ["Revenue", money(dashboard?.kpis?.revenue)],
          ["Orders", (dashboard?.kpis?.orders || 0).toLocaleString()],
          ["Customers", (dashboard?.kpis?.customers || 0).toLocaleString()],
          ["Freshness", dashboard?.freshness ? new Date(dashboard.freshness).toLocaleString() : "No data"],
        ].map(([k, v]) => (
          <article key={k}>
            <span>{k}</span>
            <strong>{v}</strong>
          </article>
        ))}
      </section>

      <section className="grid">
        <article className="panel wide">
          <p className="eyebrow">Historical sales replay</p>
          <h2>Process new events from immutable source</h2>
          <p className="sub">Choose a batch definition and replay speed. Facts are consumed after Kafka processing.</p>

          <div className="controls">
            <label>
              Batch mode
              <select value={batchMode} onChange={(e) => { setBatchMode(e.target.value); setBatchValue(""); }}>
                <option value="events">Events</option>
                <option value="day">One day</option>
                <option value="week">One week</option>
                <option value="store">One store</option>
              </select>
            </label>

            {batchMode !== "events" && (
              <label>
                {batchMode === "day" ? "Day" : batchMode === "week" ? "Week" : "Store"}
                {modeOptions.length ? (
                  <select value={batchValue} onChange={(e) => setBatchValue(e.target.value)}>
                    <option value="">Select</option>
                    {modeOptions.map((x) => (
                      <option key={x.value} value={x.value}>
                        {x.value} ({x.events.toLocaleString()} events)
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    value={batchValue}
                    onChange={(e) => setBatchValue(e.target.value)}
                    placeholder={batchMode === "day" ? "YYYY-MM-DD" : batchMode === "week" ? "YYYY-W01" : "Store SK"}
                  />
                )}
              </label>
            )}

            <label>
              Events per publish batch
              <input type="number" min="1" max="5000" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
            </label>
            <label>
              Seconds between publish batches
              <input type="number" min="0.1" step="0.1" value={intervalSeconds} onChange={(e) => setIntervalSeconds(e.target.value)} />
            </label>

            <button disabled={replay?.status === "running"} onClick={startReplay}>Start processing</button>
            {replay?.status === "running" && <button className="secondary" onClick={() => controlReplay("pause")}>Pause</button>}
            {replay?.status === "paused" && <button onClick={() => controlReplay("resume")}>Resume</button>}
            <button className="text" onClick={() => controlReplay("stop")}>Stop</button>
          </div>

          <div className="progress"><i style={{ width: `${progressPct}%` }} /></div>
          <div className="stats">
            Status <b>{replay?.status || "loading"}</b> · Published <b>{(replay?.events_published || 0).toLocaleString()}</b> ·
            Consumed <b>{(replay?.events_consumed || 0).toLocaleString()}</b> · Remaining <b>{(replay?.events_remaining || 0).toLocaleString()}</b> ·
            Failures <b>{(replay?.failed_events || 0).toLocaleString()}</b> · Rate <b>{(replay?.processing_rate_eps || 0).toLocaleString()} eps</b>
          </div>
        </article>

        <article className="panel">
          <p className="eyebrow">System health</p>
          <h2>Operations status</h2>
          <dl>
            <dt>API</dt><dd><StatusBadge label={diagnostics?.api} /></dd>
            <dt>Database</dt><dd><StatusBadge label={diagnostics?.database} /></dd>
            <dt>Kafka</dt><dd><StatusBadge label={diagnostics?.kafka} /></dd>
            <dt>Replay</dt><dd><StatusBadge label={diagnostics?.replay_status} /></dd>
            <dt>Processed events</dt><dd>{diagnostics?.processed_transactions?.toLocaleString() || 0}</dd>
            <dt>Failed events</dt><dd>{diagnostics?.failed_events?.toLocaleString() || 0}</dd>
            <dt>Data freshness</dt><dd>{diagnostics?.data_freshness ? new Date(diagnostics.data_freshness).toLocaleString() : "No data"}</dd>
          </dl>
        </article>
      </section>

      <section className="grid">
        <article className="panel wide">
          <p className="eyebrow">Revenue trend</p>
          <h2>Daily processed sales</h2>
          <div className="bars">
            {dashboard?.revenue_trend?.length ? (
              dashboard.revenue_trend.map((x) => {
                const peak = Math.max(...dashboard.revenue_trend.map((y) => y.revenue), 1);
                const pct = Math.max(5, (x.revenue / peak) * 100);
                return <div key={x.date} title={`${x.date}: ${money(x.revenue)}`} style={{ height: `${pct}%` }} />;
              })
            ) : (
              <p>Start replay to populate processed sales.</p>
            )}
          </div>
        </article>

        <article className="panel">
          <p className="eyebrow">Top products</p>
          <h2>Current leaders</h2>
          <ol>
            {dashboard?.top_products?.map((x) => (
              <li key={x.name}><span>{x.name}</span><b>{money(x.revenue)}</b></li>
            ))}
          </ol>
        </article>
      </section>

      <section className="panel">
        <p className="eyebrow">Business analysis</p>
        <h2>Explore processed transactions</h2>
        <div className="controls">
          <button onClick={() => runAnalysis("revenue_by_category")}>Revenue by category</button>
          <button className="secondary" onClick={() => runAnalysis("daily_revenue")}>Daily revenue</button>
          <button className="secondary" onClick={() => runAnalysis("top_customers")}>Top customers</button>
        </div>
        {analysis?.rows && (
          <ol>
            {analysis.rows.slice(0, 10).map((x) => (
              <li key={x.label}><span>{x.label}</span><b>{money(x.value)}</b></li>
            ))}
          </ol>
        )}
        {analysis?.empty && <p>No processed transactions are available yet.</p>}
      </section>

      <section className="grid">
        <article className="panel">
          <p className="eyebrow">Model availability</p>
          <h2>ML artifacts</h2>
          <ul className="model-list">
            {mlStatus?.models?.map((m) => (
              <li key={m.model}>
                <span>{m.model}</span>
                <StatusBadge label={m.artifact_present ? "Healthy" : "Warning"} />
              </li>
            ))}
          </ul>
          <p className="sub">{mlStatus?.note}</p>
        </article>

        <article className="panel">
          <p className="eyebrow">Excel validation</p>
          <h2>Preview without modifying source</h2>
          <p className="sub">Upload .xlsx or .xls up to 25 MB.</p>
          <input type="file" accept=".xlsx,.xls" onChange={(e) => preview(e.target.files?.[0])} />
          {upload && (
            <div className="preview">
              <b>{upload.filename}</b>
              <span>{upload.rows.toLocaleString()} rows · {upload.columns.length} columns</span>
              <span>Duplicate rows: {upload.duplicate_rows.toLocaleString()}</span>
              {upload.validation_errors?.length > 0 ? (
                <ul className="warnings">
                  {upload.validation_errors.map((x) => <li key={x}>{x}</li>)}
                </ul>
              ) : (
                <p>{upload.validation}</p>
              )}
            </div>
          )}
        </article>
      </section>
    </main>
  );
}

export default App;
