// Process lifecycle helpers for resilient MCP stdio shutdown behavior handling
// FEATURE: Runtime process lifecycle and broken-pipe detection utilities

interface ErrorWithCode {
  code?: string;
}

const BROKEN_PIPE_CODES = new Set(["EPIPE", "ERR_STREAM_DESTROYED", "ECONNRESET"]);
const DEFAULT_IDLE_TIMEOUT_MS = 15 * 60 * 1000;
const MIN_IDLE_TIMEOUT_MS = 60 * 1000;
const DEFAULT_PARENT_POLL_MS = 5 * 1000;
const MIN_PARENT_POLL_MS = 1 * 1000;

export interface CleanupOptions {
  cancelEmbeddings?: () => void;
  stopTracker: () => void;
  closeServer: () => Promise<void> | void;
  closeTransport: () => Promise<void> | void;
  stopMonitors?: () => void;
}

export interface IdleMonitor {
  touch: () => void;
  stop: () => void;
}

export interface IdleMonitorOptions {
  timeoutMs: number;
  onIdle: () => void;
  isTransportAlive?: () => boolean;
}

export interface ParentMonitorOptions {
  parentPid: number;
  pollIntervalMs?: number;
  onParentExit: () => void;
  isProcessAlive?: (pid: number) => boolean;
}

function toIntegerOr(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function unrefHandle(handle: { unref?: () => void } | null): void {
  handle?.unref?.();
}

export function isBrokenPipeError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const { code } = error as ErrorWithCode;
  return typeof code === "string" && BROKEN_PIPE_CODES.has(code);
}

export function getIdleShutdownMs(value: string | undefined): number {
  const normalized = value?.trim().toLowerCase();
  if (normalized && ["0", "false", "off", "disabled", "none"].includes(normalized)) return 0;
  return Math.max(MIN_IDLE_TIMEOUT_MS, toIntegerOr(value, DEFAULT_IDLE_TIMEOUT_MS));
}

export function getParentPollMs(value: string | undefined): number {
  return Math.max(MIN_PARENT_POLL_MS, toIntegerOr(value, DEFAULT_PARENT_POLL_MS));
}

export function isProcessAlive(pid: number, killCheck: (pid: number, signal: number) => void = process.kill): boolean {
  if (!Number.isFinite(pid) || pid <= 0) return false;

  try {
    killCheck(pid, 0);
    return true;
  } catch (error) {
    if (!error || typeof error !== "object") return false;
    const { code } = error as ErrorWithCode;
    return code !== "ESRCH";
  }
}

export function createIdleMonitor(options: IdleMonitorOptions): IdleMonitor {
  if (options.timeoutMs <= 0) {
    return {
      touch: () => { },
      stop: () => { },
    };
  }

  let timer: NodeJS.Timeout | null = null;

  const schedule = (): void => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      if (options.isTransportAlive && options.isTransportAlive()) {
        schedule();
        return;
      }
      options.onIdle();
    }, options.timeoutMs);
    unrefHandle(timer);
  };

  schedule();

  return {
    touch: schedule,
    stop: () => {
      if (!timer) return;
      clearTimeout(timer);
      timer = null;
    },
  };
}

export function startParentMonitor(options: ParentMonitorOptions): () => void {
  if (!Number.isFinite(options.parentPid) || options.parentPid <= 1 || options.parentPid === process.pid) {
    return () => { };
  }

  const pollIntervalMs = Math.max(MIN_PARENT_POLL_MS, Math.floor(options.pollIntervalMs ?? DEFAULT_PARENT_POLL_MS));
  const isAlive = options.isProcessAlive ?? isProcessAlive;
  let stopped = false;

  const stop = (): void => {
    if (stopped) return;
    stopped = true;
    clearInterval(interval);
  };

  const interval = setInterval(() => {
    if (stopped) return;
    if (process.ppid !== options.parentPid || !isAlive(options.parentPid)) {
      stop();
      options.onParentExit();
    }
  }, pollIntervalMs);

  unrefHandle(interval);
  return stop;
}

export async function runCleanup(options: CleanupOptions): Promise<void> {
  options.cancelEmbeddings?.();
  options.stopMonitors?.();
  options.stopTracker();
  await Promise.allSettled([
    Promise.resolve(options.closeServer()),
    Promise.resolve(options.closeTransport()),
  ]);
}
