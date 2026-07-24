/**
 * WorkflowsPanel — a carbon-aware workflow automation builder (n8n / Make style).
 *
 * Left:   saved workflows + node palette.
 * Center: React Flow canvas — drag nodes, wire edges, watch a run light up node
 *         by node with a live gCO2 tally.
 * Right:  config drawer for the selected node.
 *
 * Every AI node (llm / rag / agent / image / guardrail) executes through the same
 * CSS greenest-feasible router as the chat path, so the total carbon shown after
 * a run is a real receipt, not an estimate.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  fetchWorkflowNodeTypes,
  fetchWorkflows,
  fetchWorkflow,
  createWorkflow,
  updateWorkflow,
  deleteWorkflow,
  runWorkflow,
  fetchWorkflowRun,
  approveWorkflowRun,
  fetchWorkflowTemplates,
  fetchWorkflowTemplate,
  instantiateWorkflowTemplate,
} from "../lib/api";

const CATEGORY_COLOR = {
  trigger: "#7c5cff",
  ai: "#12b886",
  logic: "#f59f00",
  io: "#4dabf7",
};
const STATUS_COLOR = {
  pending: "#adb5bd",
  running: "#4dabf7",
  completed: "#12b886",
  skipped: "#868e96",
  failed: "#fa5252",
  awaiting_approval: "#f59f00",
};

let _idSeq = 1;
const newNodeId = () => `n${Date.now().toString(36)}${(_idSeq++).toString(36)}`;

// ── Custom node ──────────────────────────────────────────────────────────────
function WorkflowNode({ data, selected }) {
  const spec = data.spec || {};
  const color = CATEGORY_COLOR[spec.category] || "#868e96";
  const status = data.runStatus || "pending";
  const outs = spec.handles_out && spec.handles_out.length ? spec.handles_out : ["out"];
  return (
    <div
      style={{
        minWidth: 172,
        borderRadius: 10,
        border: `2px solid ${selected ? "#fff" : color}`,
        background: "#1b1d22",
        color: "#e9ecef",
        boxShadow: selected ? `0 0 0 2px ${color}` : "0 2px 8px rgba(0,0,0,.35)",
        fontSize: 12,
        overflow: "hidden",
      }}
    >
      {!spec.is_trigger && (
        <Handle type="target" position={Position.Left} style={{ background: color }} />
      )}
      <div style={{ background: color, padding: "5px 9px", fontWeight: 600, color: "#0b0c0f", display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{data.label || spec.label}</span>
        {data.runStatus && (
          <span title={status} style={{ width: 9, height: 9, borderRadius: "50%", background: STATUS_COLOR[status], alignSelf: "center", flexShrink: 0 }} />
        )}
      </div>
      <div style={{ padding: "6px 9px", color: "#adb5bd", fontSize: 11 }}>
        {spec.type}
        {typeof data.carbon === "number" && data.carbon > 0 && (
          <span style={{ color: "#12b886", float: "right" }}>{data.carbon.toFixed(3)} g</span>
        )}
      </div>
      {outs.map((h, i) => (
        <Handle
          key={h}
          id={h}
          type="source"
          position={Position.Right}
          style={{
            top: outs.length === 1 ? "50%" : `${28 + (i * 64) / Math.max(outs.length - 1, 1)}%`,
            background: color,
          }}
        />
      ))}
      {outs.length > 1 && (
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 6, padding: "0 9px 5px", color: "#868e96", fontSize: 10 }}>
          {outs.map((h) => <span key={h}>{h}</span>)}
        </div>
      )}
    </div>
  );
}

const nodeTypes = { wfNode: WorkflowNode };

// ── Param editors ────────────────────────────────────────────────────────────
function ParamField({ spec, value, onChange }) {
  const label = <label style={{ display: "block", fontSize: 11, color: "#adb5bd", margin: "10px 0 3px" }}>{spec.label}</label>;
  const common = { width: "100%", boxSizing: "border-box", background: "#111318", color: "#e9ecef", border: "1px solid #343a40", borderRadius: 6, padding: "6px 8px", fontSize: 12 };
  if (spec.type === "textarea") {
    return <div>{label}<textarea style={{ ...common, minHeight: 70, fontFamily: "inherit" }} value={value ?? ""} onChange={(e) => onChange(e.target.value)} /></div>;
  }
  if (spec.type === "select") {
    return <div>{label}<select style={common} value={value ?? spec.default} onChange={(e) => onChange(e.target.value)}>{(spec.options || []).map((o) => <option key={o} value={o}>{o}</option>)}</select></div>;
  }
  if (spec.type === "number") {
    return <div>{label}<input type="number" style={common} value={value ?? spec.default ?? 0} onChange={(e) => onChange(Number(e.target.value))} /></div>;
  }
  if (spec.type === "boolean") {
    return <div style={{ marginTop: 10 }}><label style={{ fontSize: 12, color: "#e9ecef", display: "flex", gap: 8, alignItems: "center" }}><input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />{spec.label}</label></div>;
  }
  if (spec.type === "keyvalue") {
    const asText = typeof value === "object" ? JSON.stringify(value, null, 2) : (value || "{}");
    return (
      <div>{label}
        <textarea
          style={{ ...common, minHeight: 70, fontFamily: "monospace" }}
          defaultValue={asText}
          onBlur={(e) => { try { onChange(JSON.parse(e.target.value || "{}")); } catch { /* keep last good */ } }}
        />
        <div style={{ fontSize: 10, color: "#868e96", marginTop: 2 }}>JSON object; values may use {"{{ $node.id.field }}"}</div>
      </div>
    );
  }
  return <div>{label}<input style={common} value={value ?? ""} onChange={(e) => onChange(e.target.value)} /></div>;
}

// ── Main panel ───────────────────────────────────────────────────────────────
function WorkflowsPanelInner() {
  const [palette, setPalette] = useState([]);
  const [workflows, setWorkflows] = useState([]);
  const [wfId, setWfId] = useState(null);
  const [name, setName] = useState("Untitled workflow");
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [run, setRun] = useState(null); // {id, status, total_carbon_g, node_states}
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [templates, setTemplates] = useState([]);
  const pollRef = useRef(null);

  const paletteByType = useMemo(() => Object.fromEntries(palette.map((p) => [p.type, p])), [palette]);

  const loadWorkflows = useCallback(async () => {
    try { setWorkflows((await fetchWorkflows()).workflows || []); } catch (e) { setError(e.message); }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setPalette((await fetchWorkflowNodeTypes()).node_types || []);
        await loadWorkflows();
      } catch (e) { setError(e.message); }
    })();
    return () => pollRef.current && clearInterval(pollRef.current);
  }, [loadWorkflows]);

  const onNodesChange = useCallback((c) => setNodes((ns) => applyNodeChanges(c, ns)), []);
  const onEdgesChange = useCallback((c) => setEdges((es) => applyEdgeChanges(c, es)), []);
  const onConnect = useCallback((c) => setEdges((es) => addEdge({ ...c, animated: true }, es)), []);
  const onSelectionChange = useCallback(({ nodes: sel }) => setSelectedId(sel && sel[0] ? sel[0].id : null), []);

  const addNode = (type) => {
    const spec = paletteByType[type];
    if (!spec) return;
    const id = newNodeId();
    const defaults = Object.fromEntries((spec.params || []).map((p) => [p.name, p.default]));
    setNodes((ns) => ns.concat({
      id,
      type: "wfNode",
      position: { x: 120 + (ns.length % 4) * 90, y: 80 + ns.length * 70 },
      data: { label: spec.label, spec, params: defaults },
    }));
    setSelectedId(id);
  };

  const updateSelectedParam = (paramName, val) => {
    setNodes((ns) => ns.map((n) => n.id === selectedId
      ? { ...n, data: { ...n.data, params: { ...n.data.params, [paramName]: val } } }
      : n));
  };
  const updateSelectedLabel = (val) => {
    setNodes((ns) => ns.map((n) => n.id === selectedId ? { ...n, data: { ...n.data, label: val } } : n));
  };
  const updateSelectedField = (field, val) => {
    setNodes((ns) => ns.map((n) => n.id === selectedId ? { ...n, data: { ...n.data, [field]: val } } : n));
  };
  const deleteSelected = () => {
    if (!selectedId) return;
    setNodes((ns) => ns.filter((n) => n.id !== selectedId));
    setEdges((es) => es.filter((e) => e.source !== selectedId && e.target !== selectedId));
    setSelectedId(null);
  };

  // ── graph <-> react-flow ──
  const toGraph = () => ({
    nodes: nodes.map((n) => ({
      id: n.id, type: n.data.spec.type, label: n.data.label,
      params: n.data.params || {}, position: n.position,
      // Per-node resilience controls (omit when default so graphs stay clean).
      ...(n.data.retries ? { retries: Number(n.data.retries) } : {}),
      ...(n.data.retry_backoff_s ? { retry_backoff_s: Number(n.data.retry_backoff_s) } : {}),
      ...(n.data.timeout_s ? { timeout_s: Number(n.data.timeout_s) } : {}),
      ...(n.data.on_error && n.data.on_error !== "stop" ? { on_error: n.data.on_error } : {}),
    })),
    edges: edges.map((e) => ({ source: e.source, target: e.target, sourceHandle: e.sourceHandle || null })),
  });

  const loadGraph = (graph) => {
    const rfNodes = (graph.nodes || []).map((n) => ({
      id: n.id, type: "wfNode",
      position: n.position || { x: 100, y: 100 },
      data: {
        label: n.label,
        spec: paletteByType[n.type] || { type: n.type, handles_out: ["out"], params: [] },
        params: n.params || {},
        retries: n.retries, retry_backoff_s: n.retry_backoff_s,
        timeout_s: n.timeout_s, on_error: n.on_error,
      },
    }));
    setNodes(rfNodes);
    setEdges((graph.edges || []).map((e, i) => ({
      id: `e${i}-${e.source}-${e.target}`, source: e.source, target: e.target,
      sourceHandle: e.sourceHandle || undefined, animated: true,
    })));
  };

  const openWorkflow = async (id) => {
    try {
      const wf = await fetchWorkflow(id);
      setWfId(wf.id); setName(wf.name); setRun(null); setSelectedId(null);
      loadGraph(wf.graph || {});
    } catch (e) { setError(e.message); }
  };

  const newWorkflow = () => {
    setWfId(null); setName("Untitled workflow"); setRun(null); setSelectedId(null);
    setNodes([]); setEdges([]);
  };

  const openGallery = async () => {
    setGalleryOpen(true);
    try { setTemplates((await fetchWorkflowTemplates()).templates || []); }
    catch (e) { setError(e.message); }
  };

  // Instantiate a template into a new editable workflow, then open it.
  const useTemplate = async (tpl) => {
    setBusy(true); setError(null);
    try {
      const wf = await instantiateWorkflowTemplate(tpl.id);
      await loadWorkflows();
      setGalleryOpen(false);
      await openWorkflow(wf.id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  // Load a template's graph onto the canvas without saving (for tweaking first).
  const previewTemplate = async (tpl) => {
    try {
      const full = await fetchWorkflowTemplate(tpl.id);
      setWfId(null); setName(full.name); setRun(null); setSelectedId(null);
      loadGraph(full.graph || {});
      setGalleryOpen(false);
    } catch (e) { setError(e.message); }
  };

  const save = async () => {
    setBusy(true); setError(null);
    try {
      const payload = { name, graph: toGraph() };
      const wf = wfId ? await updateWorkflow(wfId, payload) : await createWorkflow(payload);
      setWfId(wf.id);
      await loadWorkflows();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  const removeWorkflow = async (id) => {
    if (!window.confirm("Delete this workflow?")) return;
    try {
      await deleteWorkflow(id);
      if (id === wfId) newWorkflow();
      await loadWorkflows();
    } catch (e) { setError(e.message); }
  };

  const applyRunStates = useCallback((r) => {
    const byId = Object.fromEntries((r.node_states || []).map((s) => [s.id, s]));
    setNodes((ns) => ns.map((n) => {
      const s = byId[n.id];
      return s ? { ...n, data: { ...n.data, runStatus: s.status, carbon: s.carbon_g } } : n;
    }));
  }, []);

  const doRun = async () => {
    setBusy(true); setError(null); setRun(null);
    try {
      // Save first so the run reflects on-screen edits.
      const payload = { name, graph: toGraph() };
      const wf = wfId ? await updateWorkflow(wfId, payload) : await createWorkflow(payload);
      setWfId(wf.id);
      await loadWorkflows();
      const started = await runWorkflow(wf.id, {});
      setRun(started);
      startPolling(started.id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  const startPolling = useCallback((runId) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetchWorkflowRun(runId);
        setRun(r); applyRunStates(r);
        // Stop polling on any non-running state (terminal or awaiting approval).
        if (r.status !== "running") { clearInterval(pollRef.current); pollRef.current = null; }
      } catch { /* transient */ }
    }, 900);
  }, [applyRunStates]);

  const decide = async (nodeId, approved) => {
    if (!run) return;
    setError(null);
    try {
      await approveWorkflowRun(run.id, { nodeId, approved });
      setRun({ ...run, status: "running", awaiting: [] });
      startPolling(run.id);
    } catch (e) { setError(e.message); }
  };

  const selectedNode = nodes.find((n) => n.id === selectedId) || null;
  const selectedSpec = selectedNode ? (paletteByType[selectedNode.data.spec.type] || selectedNode.data.spec) : null;
  const grouped = useMemo(() => {
    const g = {};
    palette.forEach((p) => { (g[p.category] = g[p.category] || []).push(p); });
    return g;
  }, [palette]);

  const btn = (extra = {}) => ({ background: "#12b886", color: "#0b0c0f", border: "none", borderRadius: 8, padding: "7px 14px", fontWeight: 600, cursor: "pointer", ...extra });

  const templatesByIndustry = useMemo(() => {
    const g = {};
    templates.forEach((t) => { (g[t.industry] = g[t.industry] || []).push(t); });
    return g;
  }, [templates]);

  return (
    <div style={{ display: "flex", height: "100%", color: "#e9ecef", background: "#0e0f12", position: "relative" }}>
      {/* Template gallery modal */}
      {galleryOpen && (
        <div onClick={() => setGalleryOpen(false)}
          style={{ position: "absolute", inset: 0, zIndex: 20, background: "rgba(0,0,0,.6)",
            display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
          <div onClick={(e) => e.stopPropagation()}
            style={{ width: "min(1000px, 95%)", maxHeight: "88%", overflowY: "auto", background: "#14161b",
              border: "1px solid #23262d", borderRadius: 12, padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <div>
                <div style={{ fontSize: 17, fontWeight: 700 }}>Template gallery</div>
                <div style={{ fontSize: 12, color: "#868e96" }}>{templates.length} carbon-aware, ready-to-run workflows</div>
              </div>
              <button onClick={() => setGalleryOpen(false)} style={{ background: "transparent", color: "#adb5bd", border: "1px solid #343a40", borderRadius: 6, padding: "4px 10px", cursor: "pointer" }}>Close</button>
            </div>
            {templates.length === 0 && <div style={{ color: "#868e96", fontSize: 13, padding: 20 }}>Loading templates…</div>}
            {Object.entries(templatesByIndustry).map(([industry, items]) => (
              <div key={industry} style={{ marginTop: 16 }}>
                <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, color: "#7c5cff", marginBottom: 8 }}>{industry}</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
                  {items.map((t) => (
                    <div key={t.id} style={{ background: "#1b1d22", border: "1px solid #23262d", borderRadius: 10, padding: 12, display: "flex", flexDirection: "column", gap: 6 }}>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>{t.name}</div>
                      <div style={{ fontSize: 11.5, color: "#adb5bd", flex: 1, lineHeight: 1.4 }}>{t.description}</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                        <span style={{ fontSize: 10, color: "#868e96" }}>{t.node_count} nodes</span>
                        {(t.tags || []).map((tag) => (
                          <span key={tag} style={{ fontSize: 10, color: "#12b886", background: "#12b88618", borderRadius: 4, padding: "1px 6px" }}>{tag}</span>
                        ))}
                      </div>
                      <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                        <button disabled={busy} onClick={() => useTemplate(t)} style={btn({ flex: 1, padding: "6px 10px", fontSize: 12, opacity: busy ? 0.6 : 1 })}>Use template</button>
                        <button onClick={() => previewTemplate(t)} style={{ background: "transparent", color: "#adb5bd", border: "1px solid #343a40", borderRadius: 8, padding: "6px 10px", fontSize: 12, cursor: "pointer" }}>Preview</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Left rail: workflows + palette */}
      <aside style={{ width: 230, borderRight: "1px solid #23262d", padding: 14, overflowY: "auto", flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <strong style={{ fontSize: 13 }}>Workflows</strong>
          <button onClick={newWorkflow} style={btn({ padding: "3px 9px", fontSize: 12 })}>+ New</button>
        </div>
        <button onClick={openGallery}
          style={{ width: "100%", marginTop: 8, background: "#16181d", color: "#e9ecef",
            border: "1px solid #7c5cff55", borderRadius: 6, padding: "6px 8px", fontSize: 12, cursor: "pointer" }}>
          ✨ Browse templates
        </button>
        <div style={{ marginTop: 10 }}>
          {workflows.length === 0 && <div style={{ color: "#868e96", fontSize: 12 }}>No workflows yet.</div>}
          {workflows.map((w) => (
            <div key={w.id} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
              <button onClick={() => openWorkflow(w.id)} title={w.description}
                style={{ flex: 1, textAlign: "left", background: w.id === wfId ? "#23262d" : "transparent", color: "#e9ecef", border: "1px solid #23262d", borderRadius: 6, padding: "6px 8px", fontSize: 12, cursor: "pointer", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {w.name}
              </button>
              <button onClick={() => removeWorkflow(w.id)} title="Delete" style={{ background: "transparent", color: "#868e96", border: "none", cursor: "pointer" }}>✕</button>
            </div>
          ))}
        </div>

        <strong style={{ fontSize: 13, display: "block", margin: "18px 0 8px" }}>Add node</strong>
        {Object.entries(grouped).map(([cat, items]) => (
          <div key={cat} style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 10, textTransform: "uppercase", color: CATEGORY_COLOR[cat] || "#868e96", marginBottom: 4, letterSpacing: 0.5 }}>{cat}</div>
            {items.map((p) => (
              <button key={p.type} onClick={() => addNode(p.type)} title={p.description}
                style={{ display: "block", width: "100%", textAlign: "left", background: "#16181d", color: "#e9ecef", border: `1px solid ${CATEGORY_COLOR[cat] || "#343a40"}44`, borderRadius: 6, padding: "5px 8px", fontSize: 12, cursor: "pointer", marginBottom: 3 }}>
                {p.label}
              </button>
            ))}
          </div>
        ))}
      </aside>

      {/* Center: toolbar + canvas */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderBottom: "1px solid #23262d" }}>
          <input value={name} onChange={(e) => setName(e.target.value)}
            style={{ background: "#111318", color: "#e9ecef", border: "1px solid #343a40", borderRadius: 6, padding: "7px 10px", fontSize: 14, minWidth: 220 }} />
          <button onClick={save} disabled={busy} style={btn({ background: "#4dabf7", opacity: busy ? 0.6 : 1 })}>Save</button>
          <button onClick={doRun} disabled={busy || nodes.length === 0} style={btn({ opacity: busy || nodes.length === 0 ? 0.6 : 1 })}>▶ Run</button>
          {run && (
            <span style={{ fontSize: 12, color: STATUS_COLOR[run.status] || "#adb5bd" }}>
              {run.status} · <strong style={{ color: "#12b886" }}>{(run.total_carbon_g || 0).toFixed(4)} gCO₂</strong>
            </span>
          )}
          {error && <span style={{ color: "#fa5252", fontSize: 12 }}>{error}</span>}
        </div>

        {/* Human-in-the-loop: approval banner while the run is paused. */}
        {run && run.status === "awaiting_approval" && (run.awaiting || []).map((a) => (
          <div key={a.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 14px",
            background: "#2b2410", borderBottom: "1px solid #f59f0044" }}>
            <span style={{ fontSize: 16 }}>⏸</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#f59f00" }}>Awaiting approval · {a.label}</div>
              {a.message && <div style={{ fontSize: 12, color: "#e9ecef" }}>{a.message}</div>}
            </div>
            <button onClick={() => decide(a.id, true)} style={btn({ padding: "6px 14px" })}>Approve</button>
            <button onClick={() => decide(a.id, false)} style={btn({ background: "#fa5252", padding: "6px 14px" })}>Reject</button>
          </div>
        ))}

        <div style={{ flex: 1, minHeight: 0 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onSelectionChange={onSelectionChange}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#23262d" gap={18} />
            <Controls />
            <MiniMap pannable zoomable style={{ background: "#16181d" }} nodeColor={(n) => CATEGORY_COLOR[n.data?.spec?.category] || "#868e96"} />
          </ReactFlow>
        </div>
      </div>

      {/* Right: node config */}
      {selectedNode && selectedSpec && (
        <aside style={{ width: 280, borderLeft: "1px solid #23262d", padding: 14, overflowY: "auto", flexShrink: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <strong style={{ fontSize: 13 }}>{selectedSpec.label}</strong>
            <button onClick={deleteSelected} style={{ background: "transparent", color: "#fa5252", border: "1px solid #fa525244", borderRadius: 6, padding: "2px 8px", cursor: "pointer", fontSize: 12 }}>Delete</button>
          </div>
          <div style={{ fontSize: 11, color: "#868e96", marginTop: 4 }}>{selectedSpec.description}</div>

          <label style={{ display: "block", fontSize: 11, color: "#adb5bd", margin: "12px 0 3px" }}>Label</label>
          <input value={selectedNode.data.label || ""} onChange={(e) => updateSelectedLabel(e.target.value)}
            style={{ width: "100%", boxSizing: "border-box", background: "#111318", color: "#e9ecef", border: "1px solid #343a40", borderRadius: 6, padding: "6px 8px", fontSize: 12 }} />

          {(selectedSpec.params || []).map((p) => (
            <ParamField key={p.name} spec={p}
              value={selectedNode.data.params?.[p.name]}
              onChange={(v) => updateSelectedParam(p.name, v)} />
          ))}

          <details style={{ marginTop: 14 }}>
            <summary style={{ cursor: "pointer", fontSize: 12, color: "#adb5bd" }}>Advanced · retries / timeout / errors</summary>
            <ParamField spec={{ label: "Retries", type: "number", default: 0 }}
              value={selectedNode.data.retries} onChange={(v) => updateSelectedField("retries", v)} />
            <ParamField spec={{ label: "Retry backoff (s)", type: "number", default: 0 }}
              value={selectedNode.data.retry_backoff_s} onChange={(v) => updateSelectedField("retry_backoff_s", v)} />
            <ParamField spec={{ label: "Timeout (s, 0 = none)", type: "number", default: 0 }}
              value={selectedNode.data.timeout_s} onChange={(v) => updateSelectedField("timeout_s", v)} />
            <ParamField spec={{ label: "On error", type: "select", options: ["stop", "continue"], default: "stop" }}
              value={selectedNode.data.on_error || "stop"} onChange={(v) => updateSelectedField("on_error", v)} />
          </details>

          {selectedNode.data.runStatus && (
            <div style={{ marginTop: 14, padding: 8, background: "#16181d", borderRadius: 6, fontSize: 11 }}>
              <div>Last run: <span style={{ color: STATUS_COLOR[selectedNode.data.runStatus] }}>{selectedNode.data.runStatus}</span></div>
              {selectedNode.data.carbon > 0 && <div style={{ color: "#12b886" }}>{selectedNode.data.carbon.toFixed(4)} gCO₂</div>}
            </div>
          )}
        </aside>
      )}
    </div>
  );
}

export function WorkflowsPanel() {
  return (
    <ReactFlowProvider>
      <WorkflowsPanelInner />
    </ReactFlowProvider>
  );
}

export default WorkflowsPanel;
