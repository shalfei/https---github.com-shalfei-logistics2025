# python scoring.py   --orders ml_ozon_logistic_dataSetOrders.json   --couriers ml_ozon_logistic_dataSetCouriers.json   --durations_db durations.sqlite   --submission solution.json
#!/usr/bin/env python3
import json
import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict

WAREHOUSE_ID = 0
MAX_WORK_TIME = 12 * 3600
PENALTY_UNASSIGNED = 3000
MAX_ROUTE_LEN = 100000
BATCH = 20000
MAX_MISSING_PRINT = 10

def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def connect_ro(db_path: Path):
    uri = f"file:{db_path.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-524288;")
    conn.execute("ATTACH DATABASE ':memory:' AS tmp;")
    return conn

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = []
        try:
            for _ in range(n):
                batch.append(next(it))
        except StopIteration:
            if batch:
                yield batch
            break
        yield batch

def build_maps(orders_json, couriers_json):
    orders = {}
    mp_of = {}
    for o in orders_json.get("Orders", []):
        oid = o.get("ID")
        mp = o.get("MpId")
        if isinstance(oid, int) and isinstance(mp, int):
            orders[oid] = o
            mp_of[oid] = mp
    allowed_couriers = set()
    svc = defaultdict(dict)
    for c in couriers_json.get("Couriers", []):
        cid = c.get("ID")
        if not isinstance(cid, int):
            continue
        allowed_couriers.add(cid)
        for s in c.get("ServiceTimeInMps", []):
            mp = s.get("MpID")
            st = s.get("ServiceTime")
            if isinstance(mp, int) and isinstance(st, int) and st >= 0:
                svc[cid][mp] = st
    return orders, mp_of, allowed_couriers, svc

def service_time(svc, cid, mp):
    return svc.get(cid, {}).get(mp, 300)

def normalize_seq(seq):
    if not seq:
        return []
    s = list(seq)
    while s and s[0] == WAREHOUSE_ID:
        s.pop(0)
    while s and s[-1] == WAREHOUSE_ID:
        s.pop()
    if WAREHOUSE_ID in s:
        return None
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True)
    ap.add_argument("--couriers", required=True)
    ap.add_argument("--durations_db", required=True)
    ap.add_argument("--submission", required=True)
    args = ap.parse_args()

    orders_json = jload(args.orders)
    couriers_json = jload(args.couriers)
    subm = jload(args.submission)

    if not isinstance(subm, dict) or "routes" not in subm or not isinstance(subm["routes"], list):
        print("reason: bad submission format"); print("score=inf"); return

    orders, mp_of, allowed_couriers, svc = build_maps(orders_json, couriers_json)

    conn = connect_ro(Path(args.durations_db))
    cur = conn.cursor()

    routes_by_courier = {}
    used_orders = set()
    mp_owner = {}
    all_pairs = set()

    for r in subm["routes"]:
        if not isinstance(r, dict):
            print("reason: route item is not an object"); print("score=inf"); return
        cid = r.get("courier_id")
        seq = r.get("route")
        if not isinstance(cid, int) or cid not in allowed_couriers:
            print(f"reason: invalid courier_id={cid}"); print("score=inf"); return
        if not isinstance(seq, list) or len(seq) > MAX_ROUTE_LEN:
            print("reason: invalid route field or too long"); print("score=inf"); return
        if cid in routes_by_courier:
            print(f"reason: duplicate courier_id={cid}"); print("score=inf"); return
        if any((not isinstance(x, int)) for x in seq):
            print(f"reason: non-integer id in route of courier {cid}"); print("score=inf"); return

        norm = normalize_seq(seq)
        if norm is None:
            print(f"reason: warehouse id 0 appears in the middle of route of courier {cid}"); print("score=inf"); return
        if any((x not in orders) for x in norm):
            bad = [x for x in norm if x not in orders][:MAX_MISSING_PRINT]
            print(f"reason: unknown ids in route of courier {cid}: {bad}"); print("score=inf"); return
        if any((x in used_orders) for x in norm):
            dup = [x for x in norm if x in used_orders][:MAX_MISSING_PRINT]
            print(f"reason: duplicate orders across routes: {dup}"); print("score=inf"); return

        routes_by_courier[cid] = norm
        for x in norm:
            used_orders.add(x)
            mp = mp_of[x]
            if mp not in mp_owner:
                mp_owner[mp] = cid
            elif mp_owner[mp] != cid:
                print(f"reason: MpId {mp} split between couriers {mp_owner[mp]} and {cid}")
                print("score=inf"); return

        if norm:
            all_pairs.add((WAREHOUSE_ID, norm[0]))
            for a, b in zip(norm[:-1], norm[1:]):
                all_pairs.add((a, b))
            all_pairs.add((norm[-1], WAREHOUSE_ID))

    if not routes_by_courier:
        unassigned = len(orders)
        score = unassigned * PENALTY_UNASSIGNED
        print(f"score={int(score)}")
        return

    cur.execute("CREATE TABLE tmp.pairs(f INTEGER, t INTEGER);")
    cur.execute("CREATE TABLE tmp.got(f INTEGER, t INTEGER, d INTEGER);")

    pair_to_d = {}
    direct_hits = 0
    reverse_hits = 0
    via_wh_hits = 0
    missing_total = 0

    def fetch_pairs(pairs):
        cur.execute("DELETE FROM tmp.pairs;")
        cur.executemany("INSERT INTO tmp.pairs(f,t) VALUES(?,?);", pairs)
        cur.execute("DELETE FROM tmp.got;")
        cur.execute("INSERT INTO tmp.got SELECT p.f,p.t,d.d FROM tmp.pairs p JOIN main.dists d ON d.f=p.f AND d.t=p.t;")

    for batch in batched(iter(all_pairs), BATCH):
        fetch_pairs(batch)
        for f,t,d in cur.execute("SELECT f,t,d FROM tmp.got;"):
            pair_to_d[(f,t)] = int(d)
        direct_hits += sum(1 for p in batch if p in pair_to_d)

        missing = [p for p in batch if p not in pair_to_d]
        if missing:
            rev = [(b,a) for (a,b) in missing]
            fetch_pairs(rev)
            for f,t,d in cur.execute("SELECT f,t,d FROM tmp.got;"):
                pair_to_d[(t,f)] = int(d)
            reverse_hits += sum(1 for p in missing if p in pair_to_d)
            still = [p for p in missing if p not in pair_to_d]

            if still:
                need = []
                for a,b in still:
                    need.append((a, WAREHOUSE_ID))
                    need.append((WAREHOUSE_ID, b))
                fetch_pairs(need)
                tmp_map = {}
                for f,t,d in cur.execute("SELECT f,t,d FROM tmp.got;"):
                    tmp_map[(f,t)] = int(d)
                for a,b in still:
                    d1 = tmp_map.get((a, WAREHOUSE_ID))
                    d2 = tmp_map.get((WAREHOUSE_ID, b))
                    if d1 is not None and d2 is not None:
                        pair_to_d[(a,b)] = d1 + d2
                        via_wh_hits += 1
                    else:
                        missing_total += 1
                        if missing_total <= MAX_MISSING_PRINT:
                            print(f"reason: missing edge even via warehouse: ({a}->{b})")

    if missing_total > 0:
        print(f"reason: total missing edges after all fallbacks: {missing_total}")
        print("score=inf"); return

    total = 0
    for cid, seq in routes_by_courier.items():
        t = 0
        if seq:
            t += pair_to_d[(WAREHOUSE_ID, seq[0])]
        for a, b in zip(seq[:-1], seq[1:]):
            t += pair_to_d[(a, b)]
        for oid in seq:
            t += service_time(svc, cid, mp_of[oid])
        if seq:
            t += pair_to_d[(seq[-1], WAREHOUSE_ID)]
        if t > MAX_WORK_TIME:
            print(f"reason: route of courier {cid} exceeds 12h: {t}>{MAX_WORK_TIME}")
            print("score=inf"); return
        total += t

    unassigned = len(orders) - len(used_orders)
    total += unassigned * PENALTY_UNASSIGNED

    print(f"edges: direct={direct_hits}, reverse={reverse_hits}, via_warehouse={via_wh_hits}")
    print(f"unassigned: {unassigned}")
    print(f"score={int(total)}")

if __name__ == "__main__":
    main()
