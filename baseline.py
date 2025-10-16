#test
# python baseline.py --orders ml_ozon_logistic_dataSetOrders.json --couriers ml_ozon_logistic_dataSetCouriers.json --durations_json ml_ozon_logistic_dataDurations.json --durations_db durations.sqlite --output solution.json

#!/usr/bin/env python3
import argparse
import json
import sqlite3
import time
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import ijson
from tqdm import tqdm
import psutil

WAREHOUSE_ID = 0
MAX_WORK_TIME = 12 * 3600
PENALTY = 3000
COURIERS_TO_USE = 280
BATCH = 500000

def ram_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def build_sqlite_stream(durations_json, db_path, max_rows=0):
    size = Path(durations_json).stat().st_size
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-1048576;")
    cur.execute("DROP TABLE IF EXISTS dists;")
    cur.execute("CREATE TABLE dists (f INTEGER, t INTEGER, d INTEGER);")
    conn.commit()
    inserted = 0
    t0 = time.time()
    with Path(durations_json).open("rb") as f, tqdm(total=size, unit="B", unit_scale=True, desc="Build SQLite (read)") as pbar:
        parser = ijson.items(f, "item")
        batch = []
        last_pos = 0
        for rec in parser:
            batch.append((int(rec["from"]), int(rec["to"]), int(rec["dist"])))
            if len(batch) >= BATCH:
                cur.execute("BEGIN;")
                cur.executemany("INSERT INTO dists(f,t,d) VALUES(?,?,?)", batch)
                conn.commit()
                inserted += len(batch)
                batch.clear()
                now = f.tell()
                pbar.update(now - last_pos)
                last_pos = now
                if max_rows and inserted >= max_rows:
                    break
        if batch and (not max_rows or inserted < max_rows):
            need = max_rows - inserted if max_rows else len(batch)
            cur.execute("BEGIN;")
            cur.executemany("INSERT INTO dists(f,t,d) VALUES(?,?,?)", batch[:need])
            conn.commit()
            inserted += min(len(batch), need)
            now = f.tell()
            pbar.update(now - last_pos)
    cur.execute("CREATE INDEX idx_ft ON dists(f,t);")
    conn.commit()
    conn.close()
    tqdm.write(f"Inserted rows: {inserted}, RAM: {ram_mb():.1f} MB")
    tqdm.write(f"SQLite ready in {int(time.time()-t0)}s, RAM: {ram_mb():.1f} MB")

def connect_db(db_path):
    uri = f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-524288;")
    conn.execute("PRAGMA query_only=ON;")
    return conn

def preload_warehouse_edges(cur, relevant_ids):
    d0_to = {}
    for t,d in cur.execute("SELECT t,d FROM dists WHERE f=?;", (WAREHOUSE_ID,)):
        d0_to[int(t)] = int(d)
    d_from0 = {}
    def pair_lookup(a, b):
        row = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (a, b)).fetchone()
        if row:
            return int(row[0])
        row2 = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (b, a)).fetchone()
        if row2:
            return int(row2[0])
        return None
    for a in relevant_ids:
        if a == WAREHOUSE_ID:
            d_from0[a] = 0
            continue
        val = pair_lookup(a, WAREHOUSE_ID)
        if val is not None:
            d_from0[int(a)] = val
    return d0_to, d_from0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True)
    ap.add_argument("--couriers", required=True)
    ap.add_argument("--durations_json", required=True)
    ap.add_argument("--durations_db", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--build_rows", type=int, default=0)
    args = ap.parse_args()

    print("Load Orders...")
    orders_json = load_json(args.orders)
    orders = {o["ID"]: o for o in orders_json["Orders"]}
    print(f"Orders: {len(orders)}, RAM: {ram_mb():.1f} MB")

    print("Load Couriers...")
    couriers_json = load_json(args.couriers)
    couriers_all = [c["ID"] for c in couriers_json["Couriers"]]
    couriers = couriers_all[:COURIERS_TO_USE]
    print(f"Couriers used: {len(couriers)}/{len(couriers_all)}, RAM: {ram_mb():.1f} MB")

    db_path = Path(args.durations_db)
    if not db_path.exists():
        print("Build durations SQLite...")
        build_sqlite_stream(args.durations_json, db_path, max_rows=args.build_rows)
        print("Build durations SQLite... done")
    else:
        print(f"Use existing SQLite: {db_path}")

    print("Connect DB...")
    conn = connect_db(db_path)
    cur = conn.cursor()

    d0_to, d_from0 = preload_warehouse_edges(cur, orders.keys())
    print(f"Warehouse edges: out={len(d0_to)}, in={len(d_from0)}")

    @lru_cache(maxsize=2_000_000)
    def direct(a, b):
        if a == b:
            return 0
        row = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (a, b)).fetchone()
        if row:
            return int(row[0])
        row2 = cur.execute("SELECT d FROM dists WHERE f=? AND t=?;", (b, a)).fetchone()
        if row2:
            return int(row2[0])
        return None

    @lru_cache(maxsize=2_000_000)
    def safe_dist(a, b):
        if a == b:
            return 0
        d = direct(a, b)
        if d is not None:
            return d
        d1 = d_from0.get(a)
        d2 = d0_to.get(b)
        if d1 is not None and d2 is not None:
            return d1 + d2
        return 10_000_000

    def get_service_time(courier_id, mp_id):
        c = next((c for c in couriers_json["Couriers"] if c["ID"] == courier_id), None)
        if not c:
            return 300
        for s in c.get("ServiceTimeInMps", []):
            if s.get("MpID") == mp_id:
                st = s.get("ServiceTime", 300)
                return int(st) if isinstance(st, int) and st >= 0 else 300
        return 300

    def seq_service_time(courier_id, seq):
        s = 0
        for oid in seq:
            s += get_service_time(courier_id, orders[oid]["MpId"])
        return s

    def route_time(cid, route_orders):
        if not route_orders:
            return safe_dist(WAREHOUSE_ID, WAREHOUSE_ID)
        t = 0
        pos = WAREHOUSE_ID
        for oid in route_orders:
            t += safe_dist(pos, oid)
            t += get_service_time(cid, orders[oid]["MpId"])
            pos = oid
        t += safe_dist(pos, WAREHOUSE_ID)
        return t

    print("Build micro-polygons...")
    mp_orders = defaultdict(list)
    for oid, o in orders.items():
        mp_orders[o["MpId"]].append(oid)
    print(f"Polygons: {len(mp_orders)}, RAM: {ram_mb():.1f} MB")

    print("Order inside polygons (NN with safe_dist)...")
    @lru_cache(maxsize=200_000)
    def pair_cost(a, b):
        return safe_dist(a, b)

    def nn_sequence(ids):
        if not ids:
            return []
        start = min(ids, key=lambda x: pair_cost(WAREHOUSE_ID, x))
        seq = [start]
        remaining = set(ids)
        remaining.remove(start)
        cur_id = start
        while remaining:
            nxt = min(remaining, key=lambda x: pair_cost(cur_id, x))
            seq.append(nxt)
            remaining.remove(nxt)
            cur_id = nxt
        return seq

    mp_seq = {}
    for mp_id, ids in tqdm(mp_orders.items(), desc="Polygon routing", unit="poly"):
        mp_seq[mp_id] = nn_sequence(tuple(ids))

    print("Assign polygons to couriers (greedy append)...")
    routes_orders = {cid: [] for cid in couriers}
    cur_time = {cid: 0 for cid in couriers}
    last_node = {cid: WAREHOUSE_ID for cid in couriers}
    sorted_mps = sorted(mp_seq.items(), key=lambda kv: len(kv[1]))
    unassigned_mps = []

    for mp_id, seq in tqdm(sorted_mps, desc="Greedy assign", unit="poly"):
        if not seq:
            continue
        internal = 0
        for a, b in zip(seq[:-1], seq[1:]):
            internal += pair_cost(a, b)
        best_cid = None
        best_total = None
        for cid in couriers:
            base = cur_time[cid]
            add = 0
            if routes_orders[cid]:
                add -= pair_cost(last_node[cid], WAREHOUSE_ID)
                add += pair_cost(last_node[cid], seq[0])
                add += internal
                add += pair_cost(seq[-1], WAREHOUSE_ID)
            else:
                add += pair_cost(WAREHOUSE_ID, seq[0])
                add += internal
                add += pair_cost(seq[-1], WAREHOUSE_ID)
            add += seq_service_time(cid, seq)
            cand = base + add
            if cand <= MAX_WORK_TIME:
                if best_total is None or cand < best_total:
                    best_total = cand
                    best_cid = cid
        if best_cid is not None:
            routes_orders[best_cid].extend(seq)
            cur_time[best_cid] = best_total
            last_node[best_cid] = routes_orders[best_cid][-1]
        else:
            unassigned_mps.append(mp_id)

    if unassigned_mps:
        print("Rescue pass (best insertion)...")

    @lru_cache(maxsize=200_000)
    def internal_cost(seq_tuple):
        seq = list(seq_tuple)
        s = 0
        for a, b in zip(seq[:-1], seq[1:]):
            s += pair_cost(a, b)
        return s

    def best_insertion_delta(route, seq, cid):
        internal = internal_cost(tuple(seq))
        st = seq_service_time(cid, seq)
        if not route:
            add = pair_cost(WAREHOUSE_ID, seq[0]) + internal + pair_cost(seq[-1], WAREHOUSE_ID) + st
            return 0, add
        best_pos = 0
        best_add = None
        n = len(route)
        for i in range(n + 1):
            prev_id = WAREHOUSE_ID if i == 0 else route[i - 1]
            next_id = WAREHOUSE_ID if i == n else route[i]
            add = 0
            add -= pair_cost(prev_id, next_id)
            add += pair_cost(prev_id, seq[0])
            add += internal
            add += pair_cost(seq[-1], next_id)
            add += st
            if best_add is None or add < best_add:
                best_add = add
                best_pos = i
        return best_pos, best_add

    newly = []
    for mp_id in tqdm(unassigned_mps, desc="Rescue assign", unit="poly"):
        seq = mp_seq[mp_id]
        best = None
        best_cid = None
        best_pos = None
        for cid in couriers:
            pos, add = best_insertion_delta(tuple(routes_orders[cid]), tuple(seq), cid)
            cand = cur_time[cid] + add
            if cand <= MAX_WORK_TIME:
                if best is None or cand < best:
                    best = cand
                    best_cid = cid
                    best_pos = pos
        if best_cid is not None:
            r = routes_orders[best_cid]
            routes_orders[best_cid] = r[:best_pos] + seq + r[best_pos:]
            cur_time[best_cid] = int(best)
            newly.append(mp_id)

    unassigned_mps = [m for m in unassigned_mps if m not in newly]

    assigned_orders = sum(len(r) for r in routes_orders.values())
    total_orders = len(orders)
    remaining_orders = total_orders - assigned_orders

    print("Compute final numbers...")
    times = [int(route_time(cid, routes_orders[cid])) for cid in couriers if routes_orders[cid]]
    total_work_time = sum(times)
    penalty = remaining_orders * PENALTY
    final_score = total_work_time + penalty

    routes_with_wh = []
    cleaned = 0
    for cid, r in routes_orders.items():
        if not r:
            continue
        rr = [oid for oid in r if oid in orders and oid != WAREHOUSE_ID]
        cleaned += len(r) - len(rr)
        routes_with_wh.append({"courier_id": cid, "route": [WAREHOUSE_ID] + rr + [WAREHOUSE_ID]})

    solution = {"routes": routes_with_wh}
    Path(args.output).write_text(json.dumps(solution, indent=2), encoding="utf-8")

    print(f"Saved {args.output}")
    print(f"Assigned orders: {assigned_orders}/{total_orders}")
    print(f"Unassigned orders: {remaining_orders}")
    print(f"Cleaned entries removed: {cleaned}")
    if times:
        print(f"Min route time: {min(times)}s")
        print(f"Max route time: {max(times)}s")
        print(f"Avg route time: {int(sum(times)/len(times))}s")
    print(f"Total work time: {int(total_work_time)}s")
    print(f"Penalty: {int(penalty)}s")
    print(f"Final score: {int(final_score)}s")
    print(f"RAM: {ram_mb():.1f} MB")

if __name__ == "__main__":
    main()
