// Session-scoped short-term KV silo with silo_size LRU eviction
// FEATURE: Short-Term KV — init/peek/set/flush + optional pending map for resolve

/**
 * In-process KV slot manager mirroring PMLL memory_silo_t / ppm mcp kv-store.
 * Capacity: siloSize enforced on set(). New keys at capacity evict LRU.
 * Optional pending map supports simple in-flight resolve without full Q-promise.
 */

interface KVSlot {
  index: number;
  key: string;
  value: string;
  resolved: boolean;
  /** Strictly increasing per-store access sequence (not wall clock). */
  lastAccessed: number;
}

export type PeekResult = [boolean, string | null, number | null];

export type PeekContextResult =
  | { hit: true; value: string; index: number }
  | { hit: true; status: "pending"; promise_id: string }
  | { hit: false };

export class ShortTermKVStore {
  private _slots: Map<string, KVSlot> = new Map();
  private _pending: Map<string, string> = new Map(); // key → promise_id
  private _nextIndex = 0;
  private _accessSeq = 0;
  siloSize: number;

  constructor(siloSize: number = 256) {
    this.siloSize = Math.max(1, siloSize | 0);
  }

  private _touch(slot: KVSlot): void {
    this._accessSeq += 1;
    slot.lastAccessed = this._accessSeq;
  }

  peek(key: string): PeekResult {
    const slot = this._slots.get(key);
    if (slot !== undefined && slot.resolved) {
      this._touch(slot);
      return [true, slot.value, slot.index];
    }
    return [false, null, null];
  }

  /** Two-stage guard: KV hit → pending map → miss. */
  peekContext(key: string): PeekContextResult {
    const [hit, value, index] = this.peek(key);
    if (hit && value !== null && index !== null) {
      return { hit: true, value, index };
    }
    const promiseId = this._pending.get(key);
    if (promiseId !== undefined) {
      return { hit: true, status: "pending", promise_id: promiseId };
    }
    return { hit: false };
  }

  /**
   * Store key/value. New keys at capacity evict the LRU entry.
   * Clears any pending marker for the key.
   */
  set(key: string, value: string): number {
    this._pending.delete(key);
    const existing = this._slots.get(key);
    if (existing !== undefined) {
      existing.value = value;
      existing.resolved = true;
      this._touch(existing);
      return existing.index;
    }

    if (this._slots.size >= this.siloSize) {
      this._evictLru();
    }

    const index = this._nextIndex++;
    const slot: KVSlot = {
      index,
      key,
      value,
      resolved: true,
      lastAccessed: 0,
    };
    this._touch(slot);
    this._slots.set(key, slot);
    return index;
  }

  /** Mark key as in-flight so peek returns pending instead of miss. */
  markPending(key: string, promiseId?: string): string {
    const id = promiseId ?? key;
    if (!this._slots.has(key) || !this._slots.get(key)!.resolved) {
      this._pending.set(key, id);
    }
    return id;
  }

  /**
   * Resolve a pending promise. If payload is provided, stores it via set.
   * Returns status for the MCP resolve tool.
   */
  resolve(promiseId: string, payload?: string): { status: "resolved" | "pending"; payload: string | null } {
    for (const [key, id] of this._pending) {
      if (id === promiseId || key === promiseId) {
        if (payload !== undefined) {
          this.set(key, payload);
          return { status: "resolved", payload };
        }
        const [hit, value] = this.peek(key);
        if (hit && value !== null) {
          this._pending.delete(key);
          return { status: "resolved", payload: value };
        }
        return { status: "pending", payload: null };
      }
    }
    // Also accept resolving by already-stored key matching promiseId
    const [hit, value] = this.peek(promiseId);
    if (hit && value !== null) {
      return { status: "resolved", payload: value };
    }
    return { status: "pending", payload: null };
  }

  private _evictLru(): void {
    let victim: string | null = null;
    let oldest = Infinity;
    for (const [k, slot] of this._slots) {
      if (slot.lastAccessed < oldest) {
        oldest = slot.lastAccessed;
        victim = k;
      }
    }
    if (victim !== null) {
      this._slots.delete(victim);
      this._pending.delete(victim);
    }
  }

  flush(): number {
    const count = this._slots.size;
    this._slots.clear();
    this._pending.clear();
    this._nextIndex = 0;
    this._accessSeq = 0;
    return count;
  }

  get size(): number {
    return this._slots.size;
  }

  get pendingCount(): number {
    return this._pending.size;
  }

  has(key: string): boolean {
    return this._slots.has(key);
  }
}

const _sessionStores: Map<string, ShortTermKVStore> = new Map();

export function getStore(sessionId: string, siloSize: number = 256): ShortTermKVStore {
  let store = _sessionStores.get(sessionId);
  if (store === undefined) {
    store = new ShortTermKVStore(siloSize);
    _sessionStores.set(sessionId, store);
  }
  return store;
}

export function dropStore(sessionId: string): number {
  const store = _sessionStores.get(sessionId);
  if (store === undefined) return 0;
  const count = store.size;
  _sessionStores.delete(sessionId);
  return count;
}

/** Clear existing silo and return a fresh store (clear_on_init). */
export function resetStore(sessionId: string, siloSize: number = 256): ShortTermKVStore {
  _sessionStores.delete(sessionId);
  const store = new ShortTermKVStore(siloSize);
  _sessionStores.set(sessionId, store);
  return store;
}

export function listSessionIds(): string[] {
  return [..._sessionStores.keys()];
}

/** Test/helper access to the registry. */
export const _sessionStoresMap = _sessionStores;
