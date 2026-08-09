import { useEffect, useMemo, useState } from "react";
import type { LucideIcon } from "lucide-react";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  Archive, BookOpen, Check, CheckCircle2, ChevronRight, CircleAlert,
  FileAudio, FileImage, FileText, Folder, FolderInput, Gauge, KeyRound,
  LibraryBig, LoaderCircle, Menu, Palette, Plus, Search, Settings2,
  Sparkles, Upload, X,
} from "lucide-react";
import {
  AppStatus, LibraryItem, LibrarySnapshot,
  chooseDirectory,
  chooseFiles,
  chooseImportFolder,
  getStatus,
  importFiles,
  listenForDroppedPaths,
  saveSecret,
  searchKnowledge,
  manageLibrary,
  getBackgroundTasks,
  startLibraryTask,
} from "./bridge";

type View = "library" | "resources" | "search" | "tasks" | "settings" | "appearance";
type ColorMode = "light" | "dark";
type Task = {
  id: string;
  title: string;
  detail: string;
  state: "idle" | "running" | "done" | "error";
  output?: string;
};
type ImportFile = { path: string; name: string };
type FileKind = "folder" | "audio" | "image" | "document";

const navigation: Array<{ id: Exclude<View, "appearance">; label: string; icon: LucideIcon }> = [
  { id: "library", label: "资料库", icon: LibraryBig },
  { id: "search", label: "检索", icon: Search },
  { id: "tasks", label: "任务", icon: Gauge },
  { id: "settings", label: "设置", icon: Settings2 },
  { id: "resources", label: "资源管理", icon: FolderInput },
];

const fileIcons: Record<FileKind, LucideIcon> = {
  folder: Folder,
  audio: FileAudio,
  image: FileImage,
  document: FileText,
};

function getInitialColorMode(): ColorMode {
  return localStorage.getItem("personal-kb.color-mode") === "dark" ? "dark" : "light";
}

function fileKind(name: string): FileKind {
  if (!name.includes(".")) return "folder";
  const extension = name.split(".").pop()?.toLowerCase();
  if (["mp3", "wav", "m4a", "flac", "ogg", "opus", "aac"].includes(extension || "")) return "audio";
  if (["png", "jpg", "jpeg", "webp", "gif"].includes(extension || "")) return "image";
  return "document";
}

function StatusDot({ ready }: { ready: boolean }) {
  return <span className={`status-dot ${ready ? "ready" : "muted"}`} aria-label={ready ? "已就绪" : "未就绪"} />;
}

export default function App() {
  const [view, setView] = useState<View>("library");
  const [colorMode, setColorMode] = useState<ColorMode>(getInitialColorMode);
  const [status, setStatus] = useState<AppStatus | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [files, setFiles] = useState<ImportFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [defaultOutputDir, setDefaultOutputDir] = useState(() => localStorage.getItem("personal-kb.default-output-dir") || "");
  const [importMessage, setImportMessage] = useState("");
  const [tasks, setTasks] = useState<Task[]>([]);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [searching, setSearching] = useState(false);
  const [deepseekKey, setDeepseekKey] = useState("");
  const [dashscopeKey, setDashscopeKey] = useState("");
  const [settingsMessage, setSettingsMessage] = useState("");
  const [library, setLibrary] = useState<LibrarySnapshot | null>(null);
  const [selectedLibraryIds, setSelectedLibraryIds] = useState<string[]>([]);
  const [libraryUpdating, setLibraryUpdating] = useState(false);

  const activeTasks = useMemo(() => tasks.filter((task) => task.state === "running").length, [tasks]);
  const desktopReady = Boolean(status?.desktop && status.workspaceReady);

  useEffect(() => { void refreshStatus(); }, []);
  useEffect(() => { localStorage.setItem("personal-kb.default-output-dir", defaultOutputDir); }, [defaultOutputDir]);
  useEffect(() => { localStorage.setItem("personal-kb.color-mode", colorMode); }, [colorMode]);
  useEffect(() => { if (defaultOutputDir) void refreshLibrary(); }, [defaultOutputDir]);
  useEffect(() => { const timer = window.setInterval(() => { void getBackgroundTasks().then((remote) => setTasks(remote.map((task) => ({ ...task, detail: task.state === "running" ? "正在后台执行，可继续使用应用" : "后台任务已结束" })))); }, 1000); return () => window.clearInterval(timer); }, []);
  useEffect(() => {
    let stopListening: (() => void) | undefined;
    void listenForDroppedPaths((paths) => {
      addFiles(paths);
      setImportMessage("");
      setIsDragging(false);
    }).then((unlisten) => { stopListening = unlisten; });
    return () => stopListening?.();
  }, []);

  async function refreshStatus() {
    try {
      setStatus(await getStatus());
    } catch {
      setStatus(null);
    }
  }

  async function refreshLibrary() {
    if (!defaultOutputDir) return;
    const result = await manageLibrary(defaultOutputDir, "sync");
    if (result.ok) {
      try { setLibrary(JSON.parse(result.output) as LibrarySnapshot); } catch { setImportMessage("无法读取资料库目录。") }
    } else setImportMessage(result.error || result.output);
  }

  async function runLibraryAction(action: "process" | "recycle" | "create-note") {
    if (!defaultOutputDir || !selectedLibraryIds.length) return;
    if (action === "process") {
      const result = await startLibraryTask(defaultOutputDir, "process", selectedLibraryIds);
      setImportMessage(result.ok ? "处理任务已在后台开始。" : (result.error || result.output));
      return;
    }
    const result = await manageLibrary(defaultOutputDir, action, action === "create-note" ? { id: selectedLibraryIds[0] } : { ids: selectedLibraryIds });
    setImportMessage(result.ok ? (action === "recycle" ? "已移入知识库回收站。" : "已创建关联笔记。") : (result.error || result.output));
    await refreshLibrary();
  }

  async function linkKnowledge(relation: string, beforeId?: string, afterId?: string) {
    if (!defaultOutputDir || selectedLibraryIds.length < 2) return;
    const result = await manageLibrary(defaultOutputDir, "link", { ids: selectedLibraryIds, relation, beforeId, afterId });
    setImportMessage(result.ok ? "知识链接已保存到知识网络。" : (result.error || result.output));
  }

  async function updateLibrary() {
    if (!defaultOutputDir || libraryUpdating) return;
    setLibraryUpdating(true);
    setImportMessage("");
    try {
      const result = await startLibraryTask(defaultOutputDir, "update");
      setImportMessage(result.ok ? "更新任务已在后台开始。" : (result.error || result.output));
    } catch (error) {
      setImportMessage(`更新失败：${String(error)}`);
    } finally { setLibraryUpdating(false); }
  }

  function addFiles(paths: string[]) {
    const additions = paths.map((path) => ({ path, name: path.split(/[\\/]/).pop() || path }));
    setFiles((current) => [...current, ...additions.filter((item) => !current.some((existing) => existing.path === item.path))]);
  }

  async function pickOutputFolder() {
    const folder = await chooseDirectory("选择默认知识库目录");
    if (folder) setDefaultOutputDir(folder);
  }

  async function pickFiles() {
    addFiles(await chooseFiles());
    setImportMessage("");
  }

  async function pickImportFolder() {
    const folder = await chooseImportFolder();
    if (folder) {
      addFiles([folder]);
      setImportMessage("");
    }
  }

  function dropFiles(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const paths = Array.from(event.dataTransfer.files)
      .map((file) => (file as File & { path?: string }).path)
      .filter((path): path is string => Boolean(path));
    if (paths.length) {
      addFiles(paths);
      setImportMessage("");
      return;
    }
    setImportMessage("请在 Personal KB 桌面应用中拖入文件，或使用下方的选择按钮。");
  }

  async function startImport() {
    if (!files.length) return;
    if (!defaultOutputDir) {
      setImportMessage("请先在设置中选择默认知识库目录。");
      return;
    }
    const task: Task = {
      id: crypto.randomUUID(),
      title: "正在添加资料",
      detail: `${files.length} 个项目将写入个人知识库`,
      state: "running",
    };
    setTasks((current) => [task, ...current]);
    setView("tasks");
    try {
      const result = await importFiles(files.map((file) => file.path), defaultOutputDir);
      setTasks((current) => current.map((item) => item.id === task.id ? {
        ...item,
        state: result.ok ? "done" : "error",
        title: result.ok ? "资料已添加" : "资料添加失败",
        output: result.output || result.error,
      } : item));
      if (result.ok) setFiles([]);
    } catch (error) {
      setTasks((current) => current.map((item) => item.id === task.id ? {
        ...item,
        state: "error",
        title: "资料添加失败",
        output: String(error),
      } : item));
    }
  }

  async function submitSearch(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setAnswer("");
    try {
      const result = await searchKnowledge(query.trim(), defaultOutputDir);
      setAnswer(result.output || result.error || "没有找到匹配的资料。");
    } catch (error) {
      setAnswer(String(error));
    } finally {
      setSearching(false);
    }
  }

  async function persistKeys() {
    try {
      if (deepseekKey) await saveSecret("DEEPSEEK_API_KEY", deepseekKey);
      if (dashscopeKey) await saveSecret("DASHSCOPE_API_KEY", dashscopeKey);
      setDeepseekKey("");
      setDashscopeKey("");
      setSettingsMessage("密钥已保存到 Windows 凭据管理器。");
      await refreshStatus();
    } catch (error) {
      setSettingsMessage(`保存失败：${String(error)}`);
    }
  }

  function chooseColorMode(nextMode: ColorMode) { setColorMode(nextMode); }

  return <div className={`app-shell mode-${colorMode}`}>
    <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
      <button className="brand" onClick={() => setView("library")} aria-label="返回资料库">
        <span className="brand-mark"><BookOpen size={18} strokeWidth={2.15} /></span>
        <span className="brand-copy"><strong>Personal KB</strong><small>个人知识库</small></span>
      </button>

      <nav className="main-navigation" aria-label="主导航">
        {navigation.filter(({ id }) => id !== "settings").sort((left, right) => {
          const order = { library: 0, resources: 1, search: 2, tasks: 3, settings: 4 };
          return order[left.id] - order[right.id];
        }).map(({ id, label, icon: Icon }) => <button
          key={id}
          className={view === id ? "nav-item active" : "nav-item"}
          onClick={() => { setView(id); setMenuOpen(false); }}
        ><Icon size={17} strokeWidth={1.9} /><span>{label}</span></button>)}
      </nav>

      <div className="sidebar-lower">
        <button
          className={view === "settings" ? "nav-item sidebar-settings active" : "nav-item sidebar-settings"}
          onClick={() => { setView("settings"); setMenuOpen(false); }}
        ><Settings2 size={17} strokeWidth={1.9} /><span>设置</span></button>
        <div className="workspace-state compact">
          <StatusDot ready={desktopReady} />
          <span>{status === null ? "正在连接服务" : desktopReady ? "服务已连接" : "桌面服务未就绪"}</span>
        </div>
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <button className="icon-button mobile-menu" onClick={() => setMenuOpen(!menuOpen)} title="打开导航" aria-label="打开导航"><Menu size={20} /></button>
        <div className="breadcrumb"><span>Personal KB</span><ChevronRight size={14} /><strong>{view === "appearance" ? "外观" : navigation.find((item) => item.id === view)?.label}</strong></div>
        <div className="topbar-actions">
          {activeTasks > 0 && <span className="activity-indicator"><LoaderCircle size={14} className="spin" /> {activeTasks} 个任务</span>}
          <button className="theme-indicator" onClick={() => setView("appearance")} title="切换浅色或深色模式"><Palette size={16} /><span>{colorMode === "light" ? "浅色" : "深色"}</span></button>
          <button className="icon-button" onClick={() => void refreshStatus()} title="刷新环境状态" aria-label="刷新环境状态"><Sparkles size={17} /></button>
        </div>
      </header>

      {status !== null && !status.desktop && <div className="preview-banner"><CircleAlert size={16} /><span>当前是浏览器预览。资料导入、检索与密钥保存请在桌面应用中使用。</span></div>}

      {view === "library" && <LibraryView
        files={files}
        isDragging={isDragging}
        removeFile={(path) => setFiles((current) => current.filter((file) => file.path !== path))}
        pickFiles={() => void pickFiles()}
        pickFolder={() => void pickImportFolder()}
        dropFiles={dropFiles}
        setDragging={setIsDragging}
        onStart={() => void startImport()}
        enabled={desktopReady}
        outputConfigured={Boolean(defaultOutputDir)}
        message={importMessage}
        library={library}
        refreshLibrary={() => void refreshLibrary()}
        selectedIds={selectedLibraryIds}
        toggleSelected={(id) => setSelectedLibraryIds((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])}
        process={() => void runLibraryAction("process")}
        recycle={() => void runLibraryAction("recycle")}
        createNote={() => void runLibraryAction("create-note")}
      />}
      {view === "resources" && <ResourcesView
        configured={Boolean(defaultOutputDir)}
        message={importMessage}
        library={library}
        refreshLibrary={() => void refreshLibrary()}
        updateLibrary={() => void updateLibrary()}
        updating={libraryUpdating}
        selectedIds={selectedLibraryIds}
        toggleSelected={(id) => setSelectedLibraryIds((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])}
        process={() => void runLibraryAction("process")}
        recycle={() => void runLibraryAction("recycle")}
        createNote={() => void runLibraryAction("create-note")}
        linkKnowledge={(relation, beforeId, afterId) => void linkKnowledge(relation, beforeId, afterId)}
      />}
      {view === "search" && <SearchView query={query} setQuery={setQuery} answer={answer} searching={searching} onSearch={submitSearch} enabled={desktopReady && Boolean(defaultOutputDir)} />}
      {view === "tasks" && <TasksView tasks={tasks} onOpenLibrary={() => setView("library")} />}
      {view === "settings" && <SettingsView
        status={status}
        deepseekKey={deepseekKey}
        dashscopeKey={dashscopeKey}
        setDeepseekKey={setDeepseekKey}
        setDashscopeKey={setDashscopeKey}
        save={() => void persistKeys()}
        message={settingsMessage}
        defaultOutputDir={defaultOutputDir}
        pickOutputFolder={() => void pickOutputFolder()}
      />}
      {view === "appearance" && <AppearanceView colorMode={colorMode} onChoose={chooseColorMode} />}
    </main>
  </div>;
}

function PageHeading({ eyebrow, title, meta }: { eyebrow: string; title: string; meta?: string }) {
  return <header className="page-heading"><p className="eyebrow">{eyebrow}</p><h1>{title}</h1>{meta && <p className="page-meta">{meta}</p>}</header>;
}

function LibraryView(props: {
  files: ImportFile[];
  isDragging: boolean;
  removeFile: (path: string) => void;
  pickFiles: () => void;
  pickFolder: () => void;
  dropFiles: (event: React.DragEvent<HTMLDivElement>) => void;
  setDragging: (value: boolean) => void;
  onStart: () => void;
  enabled: boolean;
  outputConfigured: boolean;
  message: string;
  library: LibrarySnapshot | null;
  refreshLibrary: () => void;
  selectedIds: string[];
  toggleSelected: (id: string) => void;
  process: () => void;
  recycle: () => void;
  createNote: () => void;
}) {
  const hasFiles = props.files.length > 0;
  return <section className="page library-page">
    <PageHeading eyebrow="资料库" title="把资料放进你的知识库" meta={hasFiles ? `已选择 ${props.files.length} 个项目` : undefined} />
    <div className="library-workspace">
      <section
        className={`drop-board ${hasFiles ? "has-files" : ""} ${props.isDragging ? "dragging" : ""}`}
        onDragEnter={(event) => { event.preventDefault(); props.setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => { if (event.currentTarget === event.target) props.setDragging(false); }}
        onDrop={props.dropFiles}
      >
        {hasFiles ? <>
          <div className="board-header">
            <div><span className="board-kicker">待添加</span><strong>{props.files.length} 个项目</strong></div>
            <button className="primary-button" disabled={!props.enabled} onClick={props.onStart}><Upload size={16} />添加到知识库</button>
          </div>
          <div className="file-grid">{props.files.map((file) => <FileTile key={file.path} file={file} remove={() => props.removeFile(file.path)} />)}</div>
          <div className="board-footer">
            <button className="quiet-button" disabled={!props.enabled} onClick={props.pickFiles}><Plus size={16} />添加文件</button>
            <button className="quiet-button" disabled={!props.enabled} onClick={props.pickFolder}><Folder size={16} />添加文件夹</button>
            {!props.outputConfigured && <span className="inline-notice">请先在设置中选择知识库目录</span>}
          </div>
        </> : <div className="empty-drop-content">
          <span className="drop-symbol"><Upload size={27} strokeWidth={1.65} /></span>
          <strong>{props.isDragging ? "松开以添加资料" : "拖入资料"}</strong>
          <div className="drop-actions"><button className="primary-button" disabled={!props.enabled} onClick={props.pickFiles}>选择文件</button><button className="quiet-button" disabled={!props.enabled} onClick={props.pickFolder}>选择文件夹</button></div>
        </div>}
      </section>
      <FormatGuide />
    </div>
    {props.message && <p className="page-notice">{props.message}</p>}
    <section className="resource-list"><div className="resource-list-header"><div><p className="eyebrow">资源管理</p><h2>{props.outputConfigured ? `${props.library?.items.length || 0} 份资料` : "你暂未选定你的知识库"}</h2></div><button className="quiet-button" disabled={!props.outputConfigured} onClick={props.refreshLibrary}>刷新</button></div>{props.outputConfigured && props.library && <><div className="resource-actions"><button className="quiet-button" disabled={!props.selectedIds.length} onClick={props.process}>处理并索引</button><button className="quiet-button" disabled={props.selectedIds.length !== 1} onClick={props.createNote}>创建关联笔记</button><button className="quiet-button danger-button" disabled={!props.selectedIds.length} onClick={props.recycle}>移入回收站</button></div><div className="resource-table"><div className="resource-row resource-head"><span /><span>名称</span><span>文件夹</span><span>类型</span><span>状态</span></div>{props.library.items.map((item) => <ResourceRow key={item.id} item={item} selected={props.selectedIds.includes(item.id)} toggle={props.toggleSelected} />)}</div></>}</section>
  </section>;
}

function ResourceRow({ item, selected, toggle }: { item: LibraryItem; selected: boolean; toggle: (id: string) => void }) { const folder = item.relative_path.includes("/") ? item.relative_path.split("/").slice(0, -1).join("/") : "根目录"; return <label className={selected ? "resource-row selected" : "resource-row"}><input type="checkbox" checked={selected} onChange={() => toggle(item.id)} /><strong title={item.name}>{item.name}</strong><span title={folder}>{folder}</span><span>{item.kind === "pdf" ? "PDF" : item.kind === "audio" ? "音频" : item.kind === "image" ? "图片" : "文本"}</span><span className={item.processing_status === "processed" ? "resource-status ready" : "resource-status"}>{item.processing_status === "processed" ? "已处理" : item.processing_status === "needs_processing" ? "待同步" : "待处理"}</span></label>; }

function ResourcesView(props: {
  configured: boolean;
  message: string;
  library: LibrarySnapshot | null;
  refreshLibrary: () => void;
  updateLibrary: () => void;
  updating: boolean;
  selectedIds: string[];
  toggleSelected: (id: string) => void;
  process: () => void;
  recycle: () => void;
  createNote: () => void;
  linkKnowledge: (relation: string, beforeId?: string, afterId?: string) => void;
}) {
  const [folder, setFolder] = useState("/");
  const [expandedFolders, setExpandedFolders] = useState<string[]>(["/"]);
  const [sortBy, setSortBy] = useState<"name" | "imported" | "modified" | "status">("name");
  const [processingFilter, setProcessingFilter] = useState<"all" | "processed" | "pending" | "needs_processing" | "error">("all");
  const [filter, setFilter] = useState("");
  const [relation, setRelation] = useState("related");
  const [linkDialogOpen, setLinkDialogOpen] = useState(false);
  const [beforeId, setBeforeId] = useState("");
  const [afterId, setAfterId] = useState("");
  const selectedItems = (props.library?.items || []).filter((item) => props.selectedIds.includes(item.id));
  const items = useMemo(() => {
    const collator = new Intl.Collator("zh-CN", { numeric: true, sensitivity: "base" });
    return [...(props.library?.items || [])]
      .filter((item) => folder === "/" || item.relative_path === folder || item.relative_path.startsWith(`${folder}/`))
      .filter((item) => processingFilter === "all" || item.processing_status === processingFilter)
      .filter((item) => !filter || `${item.name} ${item.relative_path} ${item.tags.join(" ")}`.toLowerCase().includes(filter.toLowerCase()))
      .sort((a, b) => sortBy === "name" ? collator.compare(a.name, b.name) : sortBy === "imported" ? b.imported_at.localeCompare(a.imported_at) : sortBy === "modified" ? b.modified_at.localeCompare(a.modified_at) : a.processing_status.localeCompare(b.processing_status));
  }, [props.library, folder, sortBy, processingFilter, filter]);
  const allVisibleSelected = items.length > 0 && items.every((item) => props.selectedIds.includes(item.id));
  function toggleVisible() {
    items.forEach((item) => {
      if (props.selectedIds.includes(item.id) === allVisibleSelected) props.toggleSelected(item.id);
    });
  }
  if (!props.configured) return <section className="page resources-page"><PageHeading eyebrow="资源管理" title="你暂未选定你的知识库" /></section>;
  return <section className="page resources-page">
    <PageHeading eyebrow="资源管理" title="管理你的知识库资料" meta={props.library ? `${props.library.items.length} 份资料，按文件夹组织` : "正在读取知识库"} />
    {props.message && <p className="page-notice">{props.message}</p>}
    <section className="resource-list">
      <div className="resource-list-header"><div><h2>{items.length} 份资料</h2></div><div className="resource-header-actions"><button className="primary-button" disabled={props.updating} onClick={props.updateLibrary}>{props.updating ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />}一键更新</button><button className="quiet-button" disabled={props.updating} onClick={props.refreshLibrary}>刷新</button></div></div>
      <div className="resource-toolbar">
        <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="筛选名称、路径或标签" />
        <select value={processingFilter} onChange={(event) => setProcessingFilter(event.target.value as typeof processingFilter)}><option value="all">全部状态</option><option value="processed">已处理</option><option value="pending">待处理</option><option value="needs_processing">待同步</option><option value="error">处理失败</option></select>
        <select value={sortBy} onChange={(event) => setSortBy(event.target.value as typeof sortBy)}><option value="name">按名称</option><option value="imported">按存入时间</option><option value="modified">按修改时间</option><option value="status">按处理状态</option></select>
      </div>
      <div className="resource-actions"><button className="quiet-button" disabled={!props.selectedIds.length} onClick={props.process}>处理并索引</button><button className="quiet-button" disabled={props.selectedIds.length !== 1} onClick={props.createNote}>创建关联笔记</button><button className="quiet-button" disabled={props.selectedIds.length < 2} onClick={() => { setBeforeId(props.selectedIds[0] || ""); setAfterId(props.selectedIds[1] || ""); setLinkDialogOpen(true); }}>建立知识链接</button><button className="quiet-button danger-button" disabled={!props.selectedIds.length} onClick={props.recycle}>移入回收站</button></div>
      {linkDialogOpen && <div className="dialog-backdrop" role="presentation" onMouseDown={() => setLinkDialogOpen(false)}><section className="relation-dialog" role="dialog" aria-modal="true" aria-labelledby="relation-dialog-title" onMouseDown={(event) => event.stopPropagation()}><h2 id="relation-dialog-title">建立知识链接</h2><p>已选择 {props.selectedIds.length} 份资料。请选择它们之间的关系。</p><label>关系类型<select className="relation-select" value={relation} onChange={(event) => setRelation(event.target.value)}><option value="related">相关知识</option><option value="same-lecture">同一课堂</option><option value="sequence">前后关系</option></select></label>{relation === "sequence" && <div className="sequence-fields"><label>前面的资料<select className="relation-select" value={beforeId} onChange={(event) => setBeforeId(event.target.value)}>{selectedItems.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label><label>后面的资料<select className="relation-select" value={afterId} onChange={(event) => setAfterId(event.target.value)}>{selectedItems.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div>}<div className="dialog-actions"><button className="quiet-button" onClick={() => setLinkDialogOpen(false)}>取消</button><button className="primary-button" disabled={relation === "sequence" && (!beforeId || !afterId || beforeId === afterId)} onClick={() => { props.linkKnowledge(relation, relation === "sequence" ? beforeId : undefined, relation === "sequence" ? afterId : undefined); setLinkDialogOpen(false); }}>确认建立</button></div></section></div>}
      {props.library && <div className="resource-table"><FolderTreeRows folders={props.library.folders} selected={folder} onSelect={setFolder} expanded={expandedFolders} toggleExpanded={(value) => setExpandedFolders((current) => current.includes(value) ? current.filter((item) => item !== value) : [...current, value])} /><div className="resource-row resource-head"><label className="select-all"><input type="checkbox" checked={allVisibleSelected} onChange={toggleVisible} aria-label="全选当前列表" /></label><span>名称</span><span>文件夹</span><span>类型</span><span>状态</span></div>{items.map((item) => <ResourceRow key={item.id} item={item} selected={props.selectedIds.includes(item.id)} toggle={props.toggleSelected} />)}</div>}
    </section>
  </section>;
}

function FolderTreeRows({ folders, selected, onSelect, expanded, toggleExpanded }: { folders: string[]; selected: string; onSelect: (folder: string) => void; expanded: string[]; toggleExpanded: (folder: string) => void }) {
  const parentOf = (folder: string) => {
    if (folder === "/") return null;
    const separator = folder.lastIndexOf("/");
    return separator === -1 ? "/" : folder.slice(0, separator);
  };
  const ordered: string[] = [];
  const visit = (folder: string) => {
    ordered.push(folder);
    folders.filter((candidate) => parentOf(candidate) === folder)
      .sort((left, right) => left.localeCompare(right, "zh-CN"))
      .forEach(visit);
  };
  visit("/");
  const visible = ordered.filter((folder) => {
    if (folder === "/") return true;
    const parts = folder.split("/");
    const parents = ["/", ...parts.slice(0, -1).map((_, index) => parts.slice(0, index + 1).join("/"))];
    return parents.every((parent) => expanded.includes(parent));
  });
  return <div className="folder-tree-rows" aria-label="我的文件夹">
    {visible.map((folder) => {
      const depth = folder === "/" ? 0 : folder.split("/").length;
      const children = ordered.some((candidate) => candidate !== folder && (folder === "/" ? !candidate.includes("/") : candidate.startsWith(`${folder}/`)));
      const open = expanded.includes(folder);
      const label = folder === "/" ? "我的文件夹" : folder.split("/").pop() || folder;
      return <div key={folder} className={selected === folder ? "folder-tree-row active" : "folder-tree-row"} style={{ paddingLeft: `${14 + depth * 18}px` }}><button className={children && open ? "folder-chevron open" : "folder-chevron"} disabled={!children} onClick={() => children && toggleExpanded(folder)} aria-label={open ? "收起文件夹" : "展开文件夹"}>{children && <ChevronRight size={15} />}</button><button className="folder-select" onClick={() => onSelect(folder)}><Folder size={16} strokeWidth={1.75} /><span>{label}</span></button></div>;
    })}
  </div>;
}

function FormatGuide() {
  const formats: Array<{ icon: LucideIcon; title: string; detail: string }> = [
    { icon: FileText, title: "文档", detail: "PDF · TXT · Markdown" },
    { icon: FileAudio, title: "音频", detail: "MP3 · WAV · M4A · FLAC" },
    { icon: FileImage, title: "图片", detail: "PNG · JPG · WebP" },
    { icon: Folder, title: "文件夹", detail: "自动遍历支持的文件" },
  ];
  return <aside className="format-guide" aria-label="支持格式">
    <p className="guide-label">支持导入</p>
    {formats.map(({ icon: Icon, title, detail }) => <div className="format-row" key={title}><Icon size={17} strokeWidth={1.8} /><div><strong>{title}</strong><span>{detail}</span></div></div>)}
  </aside>;
}

function FileTile({ file, remove }: { file: ImportFile; remove: () => void }) {
  const kind = fileKind(file.name);
  const Icon = fileIcons[kind];
  return <article className={`file-tile file-${kind}`}>
    <button className="remove-file" title={`移除 ${file.name}`} aria-label={`移除 ${file.name}`} onClick={remove}><X size={13} /></button>
    <span className="file-icon"><Icon size={29} strokeWidth={1.55} /></span>
    <span className="file-name" title={file.name}>{file.name}</span>
  </article>;
}

function SearchView(props: {
  query: string;
  setQuery: (value: string) => void;
  answer: string;
  searching: boolean;
  onSearch: (event: React.FormEvent) => void;
  enabled: boolean;
}) {
  return <section className="page search-page">
    <PageHeading eyebrow="知识检索" title="在你的资料中查找答案" />
    <form className="search-box" onSubmit={props.onSearch}>
      <Search size={20} strokeWidth={1.8} />
      <input value={props.query} onChange={(event) => props.setQuery(event.target.value)} placeholder="输入问题，例如：线性变换的几何意义是什么？" />
      <button className="primary-button" disabled={!props.enabled || props.searching}>{props.searching ? <LoaderCircle className="spin" size={16} /> : "检索"}</button>
    </form>
    {props.answer ? <SearchResult content={props.answer} /> : <div className="search-empty"><Search size={25} strokeWidth={1.55} /><strong>输入一个问题开始检索</strong></div>}
  </section>;
}

function SearchResult({ content }: { content: string }) {
  const sourceMatch = content.match(/<!--PERSONAL_KB_SOURCES:(.*?)-->/s);
  const sourceData = sourceMatch ? JSON.parse(sourceMatch[1]) as string[] | { summary: string; files: string[] } : [];
  const sources = Array.isArray(sourceData) ? sourceData : sourceData.files;
  const summary = Array.isArray(sourceData) ? "课程资料" : sourceData.summary;
  const visibleContent = content.replace(/\n?<!--PERSONAL_KB_SOURCES:.*?-->/s, "").replace(/\n来源：[^\n]*\s*$/, "");
  const parts = visibleContent.split(/(\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\])/g);
  const renderText = (text: string, partIndex: number) => text
    .split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g)
    .map((fragment, index) => {
      if (fragment.startsWith("**") && fragment.endsWith("**")) {
        return <strong key={`${partIndex}-bold-${index}`}>{fragment.slice(2, -2)}</strong>;
      }
      if (fragment.startsWith("`") && fragment.endsWith("`")) {
        return <span className="inline-term" key={`${partIndex}-term-${index}`}>{fragment.slice(1, -1)}</span>;
      }
      return fragment;
    });
  return <div className="result-panel">
    {parts.map((part, index) => {
      const isInline = part.startsWith("\\(") && part.endsWith("\\)");
      const isBlock = part.startsWith("\\[") && part.endsWith("\\]");
      if (!isInline && !isBlock) return renderText(part, index);
      const math = part.slice(2, -2);
      const html = katex.renderToString(math, { displayMode: isBlock, throwOnError: false, strict: "ignore" });
      return isBlock
        ? <span className="math-block" key={index} dangerouslySetInnerHTML={{ __html: html }} />
        : <span className="math-inline" key={index} dangerouslySetInnerHTML={{ __html: html }} />;
    })}
    {sources.length > 0 && <>
      <p className="source-summary">来源：{summary}</p>
      <details className="source-details"><summary>查看具体来源</summary><ul>{sources.map((source) => <li key={source}>{source}</li>)}</ul></details>
    </>}
  </div>;
}

function TasksView({ tasks, onOpenLibrary }: { tasks: Task[]; onOpenLibrary: () => void }) {
  return <section className="page tasks-page">
    <PageHeading eyebrow="本地任务" title="资料处理进度" />
    {tasks.length === 0 ? <div className="tasks-empty"><Archive size={26} strokeWidth={1.55} /><strong>还没有处理任务</strong><button className="quiet-button" onClick={onOpenLibrary}>前往资料库</button></div> : <div className="task-list">{tasks.map((task) => <TaskRow task={task} key={task.id} />)}</div>}
  </section>;
}

function TaskRow({ task }: { task: Task }) {
  const Icon = task.state === "running" ? LoaderCircle : task.state === "done" ? CheckCircle2 : task.state === "error" ? CircleAlert : Gauge;
  const label = task.state === "running" ? "处理中" : task.state === "done" ? "已完成" : task.state === "error" ? "失败" : "等待中";
  return <article className={`task-row ${task.state}`}>
    <span className="task-icon"><Icon size={18} className={task.state === "running" ? "spin" : ""} /></span>
    <div><strong>{task.title}</strong><p>{task.detail}</p><div className={`task-progress ${task.state}`} aria-label={label}><span /></div>{task.output && <pre>{task.output}</pre>}</div>
    <span className="task-state">{label}</span>
  </article>;
}

function SettingsView(props: {
  status: AppStatus | null;
  deepseekKey: string;
  dashscopeKey: string;
  setDeepseekKey: (value: string) => void;
  setDashscopeKey: (value: string) => void;
  save: () => void;
  message: string;
  defaultOutputDir: string;
  pickOutputFolder: () => void;
}) {
  const desktop = Boolean(props.status?.desktop);
  return <section className="page settings-page">
    <PageHeading eyebrow="本机配置" title="服务与知识库位置" />
    <div className="settings-layout">
      <section className="settings-group directory-group">
        <div className="group-heading"><FolderInput size={18} /><h2>默认知识库目录</h2></div>
        <p>导入后的可索引资料将保存在这里。</p>
        <div className="path-input"><input value={props.defaultOutputDir} readOnly placeholder="选择默认知识库目录" /><button className="icon-button" disabled={!desktop} title="选择默认目录" aria-label="选择默认目录" onClick={props.pickOutputFolder}><FolderInput size={17} /></button></div>
      </section>
      <section className="settings-group credentials-group">
        <div className="group-heading"><KeyRound size={18} /><h2>API 密钥</h2></div>
        <label>DeepSeek API Key<input disabled={!desktop} type="password" value={props.deepseekKey} onChange={(event) => props.setDeepseekKey(event.target.value)} placeholder={props.status?.deepseekConfigured ? "已配置，输入新值以替换" : "sk-..."} /></label>
        <label>DashScope API Key<input disabled={!desktop} type="password" value={props.dashscopeKey} onChange={(event) => props.setDashscopeKey(event.target.value)} placeholder={props.status?.dashscopeConfigured ? "已配置，输入新值以替换" : "sk-..."} /></label>
        <button className="primary-button" disabled={!desktop} onClick={props.save}>保存密钥</button>
        {props.message && <p className="settings-message">{props.message}</p>}
      </section>
      <section className="settings-group status-group">
        <div className="group-heading"><Gauge size={18} /><h2>环境状态</h2></div>
        <div className="status-list">
          <StatusLine label="桌面运行时" value={props.status?.desktop ? "已连接" : "浏览器预览"} ready={desktop} />
          <StatusLine label="项目工作区" value={props.status?.workspaceRoot || "检查中"} ready={Boolean(props.status?.workspaceReady)} />
          <StatusLine label="Python" value={props.status?.python || "检查中"} ready={Boolean(props.status?.workspaceReady)} />
          <StatusLine label="DeepSeek" value={props.status?.deepseekConfigured ? "密钥已保存" : "未配置"} ready={Boolean(props.status?.deepseekConfigured)} />
          <StatusLine label="DashScope" value={props.status?.dashscopeConfigured ? "密钥已保存" : "未配置"} ready={Boolean(props.status?.dashscopeConfigured)} />
        </div>
      </section>
    </div>
  </section>;
}

function StatusLine({ label, value, ready }: { label: string; value: string; ready: boolean }) {
  return <div className="status-line"><StatusDot ready={ready} /><div><strong>{label}</strong><span>{value}</span></div></div>;
}

function AppearanceView({ colorMode, onChoose }: { colorMode: ColorMode; onChoose: (mode: ColorMode) => void }) {
  return <section className="page appearance-page">
    <header className="appearance-heading"><p className="eyebrow">外观</p><div className="mode-title-grid"><h1>浅色模式</h1><h1>深色模式</h1></div></header>
    <div className="appearance-grid">
      {(["light", "dark"] as ColorMode[]).map((mode) => <article className={`appearance-choice mode-preview-${mode} ${colorMode === mode ? "selected" : ""}`} key={mode}>
        <div className="preview-screen" aria-hidden="true">
          <div className="preview-bar"><i /><i /><i /><span /></div>
          <div className="preview-body"><div className="preview-rail"><b /><b /><b /></div><div className="preview-content"><span className="preview-title" /><div className="preview-board"><em /><em /><em /></div></div></div>
        </div>
        <div className="appearance-copy"><span className="scheme-label">{mode === "light" ? "LIGHT" : "DARK"}</span><h2>{mode === "light" ? "浅色模式" : "深色模式"}</h2><p>{mode === "light" ? "明亮、轻盈，适合日间整理资料。" : "降低夜间刺激，适合长时间检索与阅读。"}</p></div>
        <button className={colorMode === mode ? "select-theme selected" : "select-theme"} onClick={() => onChoose(mode)}>{colorMode === mode ? <><Check size={16} />正在使用</> : `使用${mode === "light" ? "浅色" : "深色"}模式`}</button>
      </article>)}
    </div>
  </section>;
}
