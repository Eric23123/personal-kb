use serde::Serialize;
use chrono::Local;
use std::{collections::HashMap, env, fs, net::{SocketAddr, TcpStream}, path::{Path, PathBuf}, process::{Command, Stdio}, sync::{Arc, Mutex}, time::Duration};
use tauri::State;

const KEYRING_SERVICE: &str = "Personal-KB";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AppStatus {
    desktop: bool,
    workspace_root: String,
    python: String,
    workspace_ready: bool,
    deepseek_configured: bool,
    dashscope_configured: bool,
}

#[derive(Serialize)]
struct CommandResult {
    ok: bool,
    output: String,
    error: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackgroundTask { id: String, title: String, state: String, output: String }
struct TaskStore(Arc<Mutex<HashMap<String, BackgroundTask>>>);
impl Default for TaskStore { fn default() -> Self { Self(Arc::new(Mutex::new(HashMap::new()))) } }

fn project_root() -> PathBuf {
    if let Some(value) = env::var_os("PERSONAL_KB_ROOT") {
        return PathBuf::from(value);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(".."))
}

fn has_secret(name: &str) -> bool {
    secret_value(name).is_some()
}

fn secret_value(name: &str) -> Option<String> {
    // The desktop Settings page stores the authoritative key in Windows
    // Credential Manager. Prefer it over a stale inherited environment value.
    keyring::Entry::new(KEYRING_SERVICE, name)
            .and_then(|entry| entry.get_password())
            .ok()
            .filter(|value| !value.trim().is_empty())
        .or_else(|| env::var(name).ok().filter(|value| !value.trim().is_empty()))
}

fn python_command() -> String {
    env::var("PERSONAL_KB_PYTHON").unwrap_or_else(|_| "python".to_string())
}

fn ensure_openviking_running() {
    let address: SocketAddr = "127.0.0.1:1934".parse().expect("valid loopback address");
    if TcpStream::connect_timeout(&address, Duration::from_millis(250)).is_ok() { return; }
    let root = env::var("OPEN_VIKING_ROOT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("D:\\OpenViking"));
    let executable = root.join(".venv").join("Scripts").join("openviking-server.exe");
    let config = root.join("ov.conf");
    if !executable.is_file() || !config.is_file() { return; }
    let mut command = Command::new(executable);
    command.args(["--config", &config.to_string_lossy()]).stdout(Stdio::null()).stderr(Stdio::null());
    command.env("NO_PROXY", "127.0.0.1,localhost").env("no_proxy", "127.0.0.1,localhost");
    if let Some(value) = secret_value("DASHSCOPE_API_KEY") { command.env("DASHSCOPE_API_KEY", value); }
    let _ = command.spawn();
    for _ in 0..30 {
        if TcpStream::connect_timeout(&address, Duration::from_millis(250)).is_ok() { return; }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn ensure_workspace(root: &Path) -> Result<(), String> {
    if !root.join("scripts").join("ops").join("laptop_pipeline.py").is_file() {
        return Err(format!("Personal-KB 工作区不完整：{}", root.display()));
    }
    Ok(())
}

fn run_python(root: &Path, args: Vec<String>) -> CommandResult {
    if let Err(error) = ensure_workspace(root) {
        return CommandResult { ok: false, output: String::new(), error: Some(error) };
    }
    ensure_openviking_running();
    let mut command = Command::new(python_command());
    command.args(args).current_dir(root);
    // Keep local OpenViking traffic on loopback even when a system proxy is set.
    command.env("NO_PROXY", "127.0.0.1,localhost");
    command.env("no_proxy", "127.0.0.1,localhost");
    command.env("PYTHONIOENCODING", "utf-8");
    // Prefer the bundled CUDA WhisperX model so Personal KB does not depend
    // on the separate Super Translation installation.
    command.env("PERSONAL_KB_WHISPER_DEVICE", "cuda");
    command.env("PERSONAL_KB_WHISPER_COMPUTE_TYPE", "float16");
    command.env("PERSONAL_KB_WHISPER_MODEL_PATH", root.join("models").join("faster-whisper-large-v3-turbo"));
    command.env("HF_HOME", root.join("models").join("huggingface_cache"));
    command.env("HF_HUB_CACHE", root.join("models").join("huggingface_cache").join("hub"));
    command.env("TORCH_HOME", root.join("models").join("torch"));
    for key_name in ["DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"] {
        if let Some(value) = secret_value(key_name) {
            command.env(key_name, value);
        }
    }
    let output = command.output();
    match output {
        Ok(result) => {
            let stdout = String::from_utf8_lossy(&result.stdout).to_string();
            let stderr = String::from_utf8_lossy(&result.stderr).to_string();
            CommandResult { ok: result.status.success(), output: if stdout.trim().is_empty() { stderr.clone() } else { stdout }, error: (!result.status.success()).then_some(stderr) }
        }
        Err(error) => CommandResult { ok: false, output: String::new(), error: Some(format!("无法启动 Python：{error}")) },
    }
}

fn unique_source_dir(output_dir: &Path) -> Result<PathBuf, String> {
    let inbox = output_dir.join("收件箱");
    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
    let mut sequence = 1_u32;
    let source_dir = loop {
        let candidate = inbox.join(format!("import-{timestamp}-{sequence:02}"));
        if !candidate.exists() { break candidate; }
        sequence += 1;
    };
    fs::create_dir_all(&source_dir).map_err(|error| format!("无法创建知识库源文件目录：{error}"))?;
    Ok(source_dir)
}

fn supported_source(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else { return false; };
    matches!(extension.to_ascii_lowercase().as_str(),
        "pdf" | "mp3" | "wav" | "m4a" | "flac" | "ogg" | "opus" | "aac" | "wma" |
        "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "tif" | "gif" | "webp" |
        "txt" | "md" | "markdown" | "rst" | "org" | "tex" | "csv" | "json" | "yaml" | "yml" | "xml" | "html" | "htm"
    )
}

fn copy_source(source: &Path, destination: &Path) -> Result<usize, String> {
    if source.is_file() {
        if !supported_source(source) { return Ok(0); }
        let parent = destination.parent().ok_or_else(|| "无效的暂存路径".to_string())?;
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        fs::copy(source, destination).map_err(|error| format!("无法暂存 {}：{error}", source.display()))?;
        return Ok(1);
    }
    if !source.is_dir() { return Err(format!("找不到导入项目：{}", source.display())); }
    let mut copied = 0;
    for entry in fs::read_dir(source).map_err(|error| format!("无法读取文件夹 {}：{error}", source.display()))? {
        let entry = entry.map_err(|error| error.to_string())?;
        let relative_name = entry.file_name();
        copied += copy_source(&entry.path(), &destination.join(relative_name))?;
    }
    Ok(copied)
}

#[tauri::command]
fn get_app_status() -> AppStatus {
    let root = project_root();
    let python = Command::new(python_command()).arg("--version").output()
        .map(|result| {
            let value = String::from_utf8_lossy(&result.stdout).trim().to_string();
            if value.is_empty() { String::from_utf8_lossy(&result.stderr).trim().to_string() } else { value }
        })
        .unwrap_or_else(|_| "未找到 Python".to_string());
    AppStatus {
        desktop: true,
        workspace_root: root.display().to_string(),
        python,
        workspace_ready: ensure_workspace(&root).is_ok(),
        deepseek_configured: has_secret("DEEPSEEK_API_KEY"),
        dashscope_configured: has_secret("DASHSCOPE_API_KEY"),
    }
}

#[tauri::command]
fn save_secret(name: String, value: String) -> Result<(), String> {
    if name != "DEEPSEEK_API_KEY" && name != "DASHSCOPE_API_KEY" { return Err("不允许保存该密钥名称".to_string()); }
    if value.trim().is_empty() { return Err("密钥不能为空".to_string()); }
    keyring::Entry::new(KEYRING_SERVICE, &name).map_err(|error| error.to_string())?
        .set_password(&value).map_err(|error| error.to_string())
}

#[tauri::command]
fn run_pipeline(input_dir: String, output_dir: String, course: Option<String>, lecture: Option<u32>) -> CommandResult {
    let mut args = vec!["scripts/ops/laptop_pipeline.py".to_string(), "--input-dir".to_string(), input_dir, "--output-dir".to_string(), output_dir];
    if let Some(value) = course.filter(|value| !value.trim().is_empty()) { args.extend(["--course".to_string(), value]); }
    if let Some(value) = lecture { args.extend(["--lecture".to_string(), value.to_string()]); }
    run_python(&project_root(), args)
}

#[tauri::command]
fn import_files(files: Vec<String>, output_dir: String) -> CommandResult {
    if files.is_empty() { return CommandResult { ok: false, output: String::new(), error: Some("请至少选择一个文件".to_string()) }; }
    let output_path = PathBuf::from(&output_dir);
    if output_dir.trim().is_empty() { return CommandResult { ok: false, output: String::new(), error: Some("请先在设置中选择默认知识库目录".to_string()) }; }
    let source_dir = match unique_source_dir(&output_path) { Ok(path) => path, Err(error) => return CommandResult { ok: false, output: String::new(), error: Some(error) } };
    let mut copied = 0;
    for (index, file) in files.iter().enumerate() {
        let source = PathBuf::from(file);
        let name = source.file_name().and_then(|value| value.to_str()).unwrap_or("source");
        let destination = source_dir.join(format!("{index:03}-{name}"));
        match copy_source(&source, &destination) {
            Ok(count) => copied += count,
            Err(error) => {
                let _ = fs::remove_dir_all(&source_dir);
                return CommandResult { ok: false, output: String::new(), error: Some(error) };
            }
        }
    }
    if copied == 0 {
        let _ = fs::remove_dir_all(&source_dir);
        return CommandResult { ok: false, output: String::new(), error: Some("没有找到受支持的文件格式".to_string()) };
    }
    let synced = run_python(&project_root(), vec![
        "scripts/ops/library_manager.py".to_string(), "sync".to_string(),
        "--vault".to_string(), output_dir.clone(),
    ]);
    if !synced.ok { return synced; }
    run_python(&project_root(), vec![
        "scripts/ops/library_manager.py".to_string(), "process".to_string(),
        "--vault".to_string(), output_dir,
    ])
}

#[tauri::command]
fn search_knowledge(query: String, vault: String) -> CommandResult {
    if query.trim().is_empty() { return CommandResult { ok: false, output: String::new(), error: Some("检索问题不能为空".to_string()) }; }
    if vault.trim().is_empty() { return CommandResult { ok: false, output: String::new(), error: Some("请先在设置中选择知识库目录".to_string()) }; }
    let index_path = PathBuf::from(&vault).join(".personal-kb").join("lexical-index.json");
    if !index_path.is_file() { return CommandResult { ok: false, output: String::new(), error: Some("知识库还没有可检索的资料，请先在资源管理中处理并索引文件".to_string()) }; }
    run_python(&project_root(), vec!["scripts/retrieval/answer_question.py".to_string(), query, "--index-path".to_string(), index_path.to_string_lossy().to_string()])
}

#[tauri::command]
fn manage_library(vault: String, action: String, ids: Vec<String>, tags: Vec<String>, id: Option<String>, relation: Option<String>, before_id: Option<String>, after_id: Option<String>) -> CommandResult {
    let allowed = ["sync", "process", "tags", "recycle", "restore", "create-note", "link"];
    if !allowed.contains(&action.as_str()) {
        return CommandResult { ok: false, output: String::new(), error: Some("不允许的资料库操作".to_string()) };
    }
    let mut args = vec!["scripts/ops/library_manager.py".to_string(), action, "--vault".to_string(), vault];
    if !ids.is_empty() { args.push("--ids".to_string()); args.extend(ids); }
    if !tags.is_empty() { args.push("--tags".to_string()); args.extend(tags); }
    if let Some(value) = id.filter(|value| !value.trim().is_empty()) { args.push("--id".to_string()); args.push(value); }
    if let Some(value) = relation.filter(|value| !value.trim().is_empty()) { args.push("--relation".to_string()); args.push(value); }
    if let Some(value) = before_id.filter(|value| !value.trim().is_empty()) { args.push("--before-id".to_string()); args.push(value); }
    if let Some(value) = after_id.filter(|value| !value.trim().is_empty()) { args.push("--after-id".to_string()); args.push(value); }
    run_python(&project_root(), args)
}

#[tauri::command]
fn start_library_task(vault: String, action: String, ids: Vec<String>, state: State<TaskStore>) -> CommandResult {
    if vault.trim().is_empty() { return CommandResult { ok: false, output: String::new(), error: Some("请先选择知识库目录".to_string()) }; }
    if action != "update" && action != "process" { return CommandResult { ok: false, output: String::new(), error: Some("不支持的后台任务".to_string()) }; }
    let task_id = format!("task-{}", Local::now().format("%Y%m%d%H%M%S%3f"));
    let title = if action == "update" { "正在更新知识库" } else { "正在处理并索引资料" }.to_string();
    let store = state.0.clone();
    store.lock().unwrap().insert(task_id.clone(), BackgroundTask { id: task_id.clone(), title, state: "running".to_string(), output: String::new() });
    let root = project_root(); let task_for_thread = task_id.clone();
    std::thread::spawn(move || {
        let sync = if action == "update" { Some(run_python(&root, vec!["scripts/ops/library_manager.py".to_string(), "sync".to_string(), "--vault".to_string(), vault.clone()])) } else { None };
        let result = if let Some(value) = sync { if !value.ok { value } else { run_python(&root, vec!["scripts/ops/library_manager.py".to_string(), "process".to_string(), "--vault".to_string(), vault, "--ids".to_string()].into_iter().chain(ids).collect()) } } else { run_python(&root, vec!["scripts/ops/library_manager.py".to_string(), "process".to_string(), "--vault".to_string(), vault, "--ids".to_string()].into_iter().chain(ids).collect()) };
          if let Some(task) = store.lock().unwrap().get_mut(&task_for_thread) {
              let completed = result.ok;
              task.state = if completed { "done" } else { "error" }.to_string();
              task.title = match (action.as_str(), completed) {
                  ("update", true) => "知识库更新完成".to_string(),
                  ("update", false) => "知识库更新失败".to_string(),
                  (_, true) => "资料处理完成".to_string(),
                  _ => "资料处理失败".to_string(),
              };
              task.output = if completed { result.output } else { result.error.unwrap_or(result.output) };
          }
    });
    CommandResult { ok: true, output: task_id, error: None }
}

#[tauri::command]
fn get_background_tasks(state: State<TaskStore>) -> Vec<BackgroundTask> { state.0.lock().unwrap().values().cloned().collect() }

pub fn run() {
    tauri::Builder::default()
        .manage(TaskStore::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![get_app_status, save_secret, run_pipeline, import_files, search_knowledge, manage_library, start_library_task, get_background_tasks])
        .run(tauri::generate_context!())
        .expect("error while running Personal KB desktop application");
}
