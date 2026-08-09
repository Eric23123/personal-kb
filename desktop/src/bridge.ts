export type AppStatus = {
  desktop: boolean;
  workspaceRoot: string;
  python: string;
  workspaceReady: boolean;
  deepseekConfigured: boolean;
  dashscopeConfigured: boolean;
  obsidianVault?: string;
};

export type CommandResult = { ok: boolean; output: string; error?: string };
export type BackgroundTask = { id: string; title: string; state: "running" | "done" | "error"; output: string };
export type LibraryItem = { id: string; relative_path: string; name: string; kind: string; size: number; modified_at: string; imported_at: string; tags: string[]; processing_status: string; index_status: string; artifact_path?: string | null };
export type LibrarySnapshot = { vault: string; items: LibraryItem[]; folders: string[]; missing: number };

const isDesktop = () => "__TAURI_INTERNALS__" in window;

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isDesktop()) {
    throw new Error("此操作需要在 Personal KB 桌面应用中运行，浏览器预览不支持本地命令或凭据保存。");
  }
  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
  return tauriInvoke<T>(command, args);
}

export async function getStatus(): Promise<AppStatus> {
  if (!isDesktop()) {
    return {
      desktop: false,
      workspaceRoot: "D:\\Personal-kb",
      python: "浏览器预览模式",
      workspaceReady: false,
      deepseekConfigured: false,
      dashscopeConfigured: false,
    };
  }
  return invoke<AppStatus>("get_app_status");
}

export async function chooseDirectory(title: string): Promise<string | null> {
  if (!isDesktop()) return null;
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({ directory: true, multiple: false, title });
  return typeof selected === "string" ? selected : null;
}

export async function chooseFiles(): Promise<string[]> {
  if (!isDesktop()) return [];
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: true,
    title: "选择要添加到知识库的资料",
    filters: [{ name: "支持的资料", extensions: ["pdf", "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "txt", "md", "png", "jpg", "jpeg", "webp"] }],
  });
  if (typeof selected === "string") return [selected];
  return selected || [];
}

export async function chooseImportFolder(): Promise<string | null> {
  return chooseDirectory("选择要导入的资料文件夹");
}

export async function listenForDroppedPaths(onDrop: (paths: string[]) => void): Promise<() => void> {
  if (!isDesktop()) return () => undefined;
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  return getCurrentWindow().onDragDropEvent((event) => {
    if (event.payload.type === "drop") onDrop(event.payload.paths);
  });
}

export async function importFiles(files: string[], outputDir: string): Promise<CommandResult> {
  return invoke<CommandResult>("import_files", { files, outputDir });
}

export async function searchKnowledge(query: string, vault: string): Promise<CommandResult> {
  return invoke<CommandResult>("search_knowledge", { query, vault });
}

export async function saveSecret(name: "DEEPSEEK_API_KEY" | "DASHSCOPE_API_KEY", value: string): Promise<void> {
  return invoke("save_secret", { name, value });
}

export async function manageLibrary(vault: string, action: "sync" | "process" | "tags" | "recycle" | "restore" | "create-note" | "link", options: { ids?: string[]; tags?: string[]; id?: string; relation?: string; beforeId?: string; afterId?: string } = {}): Promise<CommandResult> {
  return invoke<CommandResult>("manage_library", { vault, action, ids: options.ids || [], tags: options.tags || [], id: options.id || null, relation: options.relation || "related", beforeId: options.beforeId || null, afterId: options.afterId || null });
}
export async function startLibraryTask(vault: string, action: "update" | "process", ids: string[] = []): Promise<CommandResult> { return invoke<CommandResult>("start_library_task", { vault, action, ids }); }
export async function getBackgroundTasks(): Promise<BackgroundTask[]> { return invoke<BackgroundTask[]>("get_background_tasks"); }
