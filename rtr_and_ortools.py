# python rtr_and_ortools.py --orders ml_ozon_logistic_dataSetOrders.json --couriers ml_ozon_logistic_dataSetCouriers.json --durations_json ml_ozon_logistic_dataDurations.json --durations_db durations.sqlite --output solution.json
#
import gc
import pandas as pd
import numpy as np
import random
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
import json
from ast import literal_eval
import sys
import time

from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_record_to_record
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

random.seed(42)

def ram_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

WAREHOUSE_ID = 0
MAX_WORK_TIME = 12 * 3600
PENALTY = 3000
COURIERS_TO_USE = 280
BATCH = 500000

max_tierations_internal = 1000
max_tierations_out = 600

print('max_tierations_internal = ', max_tierations_internal)
print('max_tierations_out = ', max_tierations_out)

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
    orders_df = pd.DataFrame(pd.read_json(args.orders)['Orders'].astype(str).map(literal_eval).to_list())
    orders_df.set_index(orders_df['ID'], inplace=True)
    #print("The size of orders_df {} bytes".format(sys.getsizeof(orders_df))) 

    print("Load Couriers...")
    with open(args.couriers, "r") as my_file:
        couriers_json = my_file.read()
    couriers = json.loads(couriers_json)
    couriers_ids = [i['ID'] for i in couriers['Couriers']]
    courier_time_limit = couriers['CourierTimeWork']['TSEnd'] - couriers['CourierTimeWork']['TSStart']
    #print("The size of couriers {} bytes".format(sys.getsizeof(couriers))) 

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

#расстояние от заказа до заказа если его нет в списке (гоняют через склад) если ничего не нашли то возвращают 10000 сек 0_о
    no_distance_list = []
    @lru_cache(maxsize=2_000_000)
    def safe_dist(a, b):
        if a == b:
            return 0
        d = direct(a, b)
        if d is not None:
            return d
        d1 = direct(a, WAREHOUSE_ID)
        d2 = direct(WAREHOUSE_ID, b)
        if d1 is not None and d2 is not None:
            return d1 + d2
        no_distance_list.append([a, b])
        return 10_000_000

    db_path = Path(args.durations_db)
    if not db_path.exists():
        print("Build durations SQLite...")
        build_sqlite_stream('ml_ozon_logistic_dataDurations.json', db_path, max_rows=0)
        print("Build durations SQLite... done")
    else:
        print(f"Use existing SQLite: {db_path}")


    print("Connect DB...")
    conn = connect_db(db_path)
    cur = conn.cursor()

    print("Calculate wh dist for orders...")
    orders_df['wh_dist'] = orders_df['ID'].apply(lambda x: safe_dist(x, WAREHOUSE_ID))
    print("Sort orders by wh dist...")
    orders_df = orders_df.sort_values(by=['MpId', 'wh_dist'], ascending=[True, True])

    print("Build MPL df...")
    mlp_ids_df = pd.DataFrame({'ID': orders_df['MpId'].unique()}, index = orders_df['MpId'].unique())
    mlp_ids_df['Order_list'] = orders_df.groupby('MpId')['ID'].apply(list)
    mlp_ids_df['Order_qty'] = mlp_ids_df['Order_list'].apply(lambda x: len(x))
    mlp_ids_df['Order_qty'] = mlp_ids_df['Order_qty'].astype(np.int64)
    mlp_ids_df['internal_route'] = [np.array([]) for x in range(len(mlp_ids_df))]
    mlp_ids_df['internal_time'] = 0
    mlp_ids_df['internal_time'] = mlp_ids_df['internal_time'].astype(np.int32)
    #mlp_ids_df = mlp_ids_df.drop(0, axis = 0)

    print("Build service_time_matrix...")
    service_time_matrix = pd.DataFrame(columns = couriers_ids, index = mlp_ids_df['ID'], dtype = int, data = 0)


    def get_service_time(courier_id, mp_id):
        c = next((c for c in couriers["Couriers"] if c["ID"] == courier_id), None)
        if not c:
            return 300
        for s in c.get("ServiceTimeInMps", []):
            if s.get("MpID") == mp_id:
                st = s.get("ServiceTime", 300)
                return int(st) if isinstance(st, int) and st >= 0 else 300
        return 300

    for courier_id in tqdm(service_time_matrix.columns):
    	for index, row in service_time_matrix.iterrows():
        	row[courier_id] = get_service_time(courier_id, index) 

    #print("The size of service_time_matrix {} bytes".format(sys.getsizeof(couriers))) 
    #print(f"RAM: {ram_mb():.1f} MB")

    gc.collect()

    mlp_ids_df['max_service_time'] = service_time_matrix.max(axis = 1)
    mlp_ids_df['total_max_service_time'] = mlp_ids_df['max_service_time'] * mlp_ids_df['Order_qty']

    def build_route_by_index(order_list, index_list):
        return [order_list[i] for i in index_list]    

    def build_distance_matrix(order_seq:list):
        distance_matrix = np.zeros([len(order_seq), len(order_seq)], dtype = np.int32)
        for index_from, order_from in enumerate(order_seq):
            for index_to, order_to in enumerate(order_seq):
                distance_matrix[index_from][index_to] = safe_dist(order_from, order_to)
        return distance_matrix

    print("Calculate internal routes...")
    for mpl_id in tqdm(mlp_ids_df.index, desc="Calculate internal routes", unit="poly"):
        if mlp_ids_df.loc[mpl_id, 'Order_qty'] <= 1:
            mlp_ids_df.at[mpl_id, 'internal_route'] =  mlp_ids_df.loc[mpl_id, 'Order_list']
            print(mpl_id, mlp_ids_df.loc[mpl_id, 'Order_list'])
            continue
        distance_matrix = build_distance_matrix(mlp_ids_df['Order_list'][mpl_id])
        distance_matrix[:, 0] = 0
        permutation, distance = solve_tsp_record_to_record(distance_matrix = distance_matrix, max_iterations = max_tierations_internal)    #300 #500   
        mlp_ids_df.at[mpl_id, 'internal_route'] = build_route_by_index(mlp_ids_df.loc[mpl_id, 'Order_list'], permutation)
        mlp_ids_df.loc[mpl_id, 'internal_time'] = distance
       

    mlp_ids_df['internal_start'] = mlp_ids_df['internal_route'].apply(lambda x: x[0])
    mlp_ids_df['internal_end'] = mlp_ids_df['internal_route'].apply(lambda x: x[-1])
    mlp_ids_df['from_wh'] = orders_df.loc[mlp_ids_df['internal_start'], 'wh_dist'].values
    mlp_ids_df['to_wh'] = orders_df.loc[mlp_ids_df['internal_end'], 'wh_dist'].values
    mlp_ids_df = mlp_ids_df.sort_values(by=['to_wh', 'from_wh'], ascending=[True, True])

    def route_time(route_orders):
        time_counter = 0
        position = route_orders[0]
        for order_id in route_orders:
            time_counter += safe_dist(position, order_id)
            position = order_id
        return time_counter

    mlp_ids_df['internal_time'] = mlp_ids_df['internal_route'].apply(lambda x: route_time(x))
    total_internal_time = mlp_ids_df['internal_time'].sum()
    #print('Total internal time:', total_internal_time)

    print("Building distance_matrix...")
    distance_matrix = np.zeros([len(mlp_ids_df), len(mlp_ids_df)], dtype = np.int32)
    print(distance_matrix.shape)

    for index_from, mpl_id_1 in enumerate(tqdm(mlp_ids_df.index)):
        order_from = int(mlp_ids_df.loc[mpl_id_1, 'internal_end'])
        st = mlp_ids_df.loc[mpl_id_1, 'total_max_service_time']
        int_time = mlp_ids_df.loc[mpl_id_1, 'internal_time']
        for index_to, mpl_id_2 in enumerate(mlp_ids_df.index):
                order_to = int(mlp_ids_df.loc[mpl_id_2, 'internal_start'])
                x = safe_dist(order_from, order_to)
                distance_matrix[index_from][index_to] = x + st + int_time   

    #print(f"RAM: {ram_mb():.1f} MB")

    print("Calculate outline route... (~10 min)")
    time_windows = [(0, MAX_WORK_TIME) for i in range(distance_matrix.shape[0])]

    def create_data_model():
        """Stores the data for the problem."""
        data = {}
        data["time_matrix"] = distance_matrix
        data["time_windows"] = time_windows
        data["num_vehicles"] = COURIERS_TO_USE
        data["depot"] = WAREHOUSE_ID
        return data

    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data["time_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        MAX_WORK_TIME,  # allow waiting time
        MAX_WORK_TIME,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time,
    )

    time_dimension = routing.GetDimensionOrDie(time)

    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx == data["depot"]:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    depot_idx = data["depot"]
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1])
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    print(f"Objective: {solution.ObjectiveValue()}")

    mpl_index_routes = []
    mpl_grop_times = []    
    time_dimension = routing.GetDimensionOrDie("Time")
    total_time = 0
    for vehicle_id in range(data["num_vehicles"]):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        route_list = []
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            route_list.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route_list.append(manager.IndexToNode(index))
        mpl_index_routes.append(route_list)
        mpl_grop_times.append(solution.Min(time_var))


    print('Builded', len(mpl_index_routes), 'groups')

    mpl_list =  mlp_ids_df.index.tolist()
    mpl_routes = [build_route_by_index(mpl_list, mpl_index_routes[i]) for i in range(len(mpl_index_routes))]

    print('max =', max(mpl_grop_times))
    print('min =', min(mpl_grop_times))


    mpl_grops_df = pd.DataFrame(zip(mpl_routes, mpl_grop_times), columns = ['group_route', 'time'])
    mpl_grops_df['group'] = mpl_grops_df['group_route'].apply(lambda x: x[1:-1])
    mpl_grops_df.loc[0, 'group'].insert(0,0)
    mpl_grops_df['mpl_qty'] = mpl_grops_df['group'].apply(lambda x: len(x))
    mpl_grops_df['order_qty'] = mpl_grops_df['group'].apply(lambda x: mlp_ids_df.loc[x,'Order_qty'].sum())

    print('orders in groups:', mpl_grops_df['order_qty'].sum())
    print('mpl in groups:', mpl_grops_df['mpl_qty'].sum())

    print("Build group_courier_matrix...")
    mpl_grops_df = mpl_grops_df.sort_values(by='order_qty', ascending=False)
    total_service_time_matrix = service_time_matrix.iloc[:, 0:280].apply(lambda x: x*mlp_ids_df['Order_qty'])
    mpl_grops_service_time_matrix = mpl_grops_df['group']
    group_courier_matrix = mpl_grops_df['group'].apply(lambda x: total_service_time_matrix.loc[x].sum())
    #print(f"RAM: {ram_mb():.1f} MB")

    mpl_grops_df['min_service_time'] = group_courier_matrix.min(axis = 1)

    print("Assigning best couriers...")

    groups_best_couriers = {}
    for group, couriers in group_courier_matrix.T.items():
        groups_best_couriers[group] = []
        for c_id, c_st in couriers.items():
            if c_st == mpl_grops_df.loc[group, 'min_service_time']:
                groups_best_couriers[group].append(c_id)

    mpl_grops_df['id'] = mpl_grops_df.index
    mpl_grops_df['best_couriers'] = mpl_grops_df['id'].apply(lambda x: groups_best_couriers[x].copy()) 
    mpl_grops_df['best_couriers_qty'] = mpl_grops_df['best_couriers'].apply(lambda x: len(x)) 
    mpl_grops_df['assigned_courier'] = None

    mpl_grops_df = mpl_grops_df.sort_values(by=['best_couriers_qty', 'order_qty' ], ascending=[True, False])

    def delete_from_list(arr, val):
        arr.remove(val)
        return arr
    
    for group in mpl_grops_df['id']:
        if len(mpl_grops_df.loc[group, 'best_couriers']) > 0:
            assigned_courier = mpl_grops_df.loc[group, 'best_couriers'][0]
            mpl_grops_df.loc[group, 'assigned_courier'] = assigned_courier
            mpl_grops_df['best_couriers'] = mpl_grops_df['best_couriers'].apply(lambda x: delete_from_list(x, assigned_courier) if x is not None and assigned_courier in x else x)
            mpl_grops_df['best_couriers_qty'] = mpl_grops_df['best_couriers'].apply(lambda x: len(x) if x is not None else 0)

    #print(mpl_grops_df.info())  
    
    mpl_unasigned_grops_df = mpl_grops_df.loc[(mpl_grops_df['assigned_courier'].isnull()) & (mpl_grops_df['best_couriers_qty'] == 0)].copy()
    assigned_couriers = mpl_grops_df.loc[mpl_grops_df['assigned_courier'].notnull(), 'assigned_courier'].values
    assigned_groups = mpl_grops_df.loc[mpl_grops_df['assigned_courier'].notnull(), 'id'].values

    while len(mpl_unasigned_grops_df) != 0:
        group_courier_matrix_temp = group_courier_matrix.drop(assigned_couriers, axis = 1)
        group_courier_matrix_temp = group_courier_matrix_temp.drop(assigned_groups, axis = 0)
        min_service_time_seq = group_courier_matrix_temp.min(axis = 1)
        groups_best_couriers_temp = {}
        for group, couriers in group_courier_matrix_temp.T.items():
            groups_best_couriers_temp[group] = []
            for c_id, c_st in couriers.items():
                if c_st == min_service_time_seq[group]:
                    groups_best_couriers_temp[group].append(c_id)
        mpl_unasigned_grops_df['best_couriers'] = mpl_unasigned_grops_df['id'].apply(lambda x: groups_best_couriers_temp[x].copy()) 
        mpl_unasigned_grops_df['best_couriers_qty'] = mpl_unasigned_grops_df['best_couriers'].apply(lambda x: len(x)) 
        for group in mpl_unasigned_grops_df['id']:
            if len(mpl_unasigned_grops_df.loc[group, 'best_couriers']) > 0:
                assigned_courier = mpl_unasigned_grops_df.loc[group, 'best_couriers'][0]
                mpl_unasigned_grops_df.loc[group, 'assigned_courier'] = assigned_courier
                mpl_unasigned_grops_df['best_couriers'] = mpl_unasigned_grops_df['best_couriers'].apply(lambda x: delete_from_list(x, assigned_courier) if x is not None and assigned_courier in x else x)
                mpl_unasigned_grops_df['best_couriers_qty'] = mpl_unasigned_grops_df['best_couriers'].apply(lambda x: len(x) if x is not None else 0)
        mpl_grops_df.loc[(mpl_grops_df['assigned_courier'].isnull()) & (mpl_grops_df['best_couriers_qty'] == 0), 'assigned_courier'] = mpl_unasigned_grops_df['assigned_courier']
        mpl_unasigned_grops_df = mpl_grops_df.loc[(mpl_grops_df['assigned_courier'].isnull()) & (mpl_grops_df['best_couriers_qty'] == 0)].copy()
        assigned_couriers = mpl_grops_df.loc[mpl_grops_df['assigned_courier'].notnull(), 'assigned_courier'].values
        assigned_groups = mpl_grops_df.loc[mpl_grops_df['assigned_courier'].notnull(), 'id'].values      


    #print(mpl_grops_df.info())

    def order_list_for_group(group):
        orders_group = []
        for mpl in group: 
            orders_group += mlp_ids_df.loc[mpl, 'internal_route']
        return orders_group

    mpl_grops_df['orders_group'] = mpl_grops_df['group'].apply(lambda x: order_list_for_group(x))
    mpls_couriers_df = pd.DataFrame(mpl_grops_df.loc[:, 'group'].values, columns = ['group'], index = mpl_grops_df['assigned_courier'].values)
    mpls_couriers_dict = mpls_couriers_df['group'].to_dict()

    orders_couriers_dict = {}
    for courier, group in mpls_couriers_dict.items():
        orders = []
        for mpl in group:
            orders += mlp_ids_df.loc[mpl, 'internal_route']
        orders_couriers_dict[courier] = orders 

    #print("orders_couriers_dict len:", len(orders_couriers_dict) )     
    assigned_orders = sum(len(r) for r in orders_couriers_dict.values())
    print("assigned_orders:", assigned_orders )

    for route in orders_couriers_dict.values():
        route.insert(0, WAREHOUSE_ID)
        route.append(WAREHOUSE_ID)

    routes_with_wh = {}
    routes = []   

    for courier, route in orders_couriers_dict.items():
        routes_with_wh['courier_id'] = courier
        routes_with_wh['route'] = route
        routes.append(routes_with_wh)
        routes_with_wh = {}
    solution = {"routes": routes}
    Path(args.output).write_text(json.dumps(solution, indent=2), encoding="utf-8")    

    print("That's all folks!")     

if __name__ == "__main__":
    main()


    


