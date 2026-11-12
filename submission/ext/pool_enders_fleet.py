import math

CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
BOARD_SIZE = 100.0


def _dist(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def _fleet_speed(ships):
    if ships <= 0:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * ((math.log(ships) / math.log(1000.0)) ** 1.5)


def _sun_clear(x1, y1, x2, y2):
    px, py = CENTER, CENTER
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return _dist(x1, y1, px, py) >= SUN_RADIUS
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    projx = x1 + t * dx
    projy = y1 + t * dy
    return _dist(px, py, projx, projy) >= SUN_RADIUS


def _predict_planet_pos(p, initial_by_id, angular_velocity, current_step, future_turns):
    pid = p[0]
    init = initial_by_id.get(pid)
    if init is None:
        return p[2], p[3]
    ox = init[2] - CENTER
    oy = init[3] - CENTER
    orbital_r = math.hypot(ox, oy)
    if orbital_r + init[4] >= ROTATION_RADIUS_LIMIT:
        return p[2], p[3]
    angle0 = math.atan2(oy, ox)
    new_angle = angle0 + angular_velocity * (current_step + future_turns)
    nx = CENTER + orbital_r * math.cos(new_angle)
    ny = CENTER + orbital_r * math.sin(new_angle)
    return nx, ny


def agent(obs):
    player = obs.get("player", 0)
    planets = obs.get("planets", [])
    initial_planets = obs.get("initial_planets", [])
    angular_velocity = obs.get("angular_velocity", 0.0)
    comets_data = obs.get("comets", [])
    comet_planet_ids = set(obs.get("comet_planet_ids", []))
    step = obs.get("step", 0)

    initial_by_id = {p[0]: p for p in initial_planets}
    planet_by_id = {p[0]: p for p in planets}

    my_planets = [p for p in planets if p[1] == player]
    if not my_planets:
        return []

    remaining_steps = 500 - step

    # Build comet intercept tables: pid -> list of (offset, x, y)
    comet_future = {}
    for group in comets_data:
        path_index = group.get("path_index", 0)
        for idx, pid in enumerate(group["planet_ids"]):
            path = group["paths"][idx]
            fut = []
            max_off = min(len(path) - path_index, 120)
            for off in range(1, max_off):
                fut.append((off, path[path_index + off][0], path[path_index + off][1]))
            comet_future[pid] = fut

    def reserve(p):
        _, _, x, y, radius, ships, prod = p
        r = max(2, 6 - step // 20)
        if len(my_planets) == 1:
            r = max(r, ships // 2)
        if prod >= 4:
            r += 4
        for op in planets:
            if op[1] not in (-1, player):
                if _dist(x, y, op[2], op[3]) < 22:
                    r += 4
                    break
        return min(r, ships - 1) if ships > 1 else ships

    def eval_comet(mp, t, fut):
        mp_x, mp_y, avail = mp[2], mp[3], mp[5] - reserve(mp)
        if avail < 3:
            return None
        best = None
        checked = 0
        for off, tx, ty in fut:
            if checked > 50:
                break
            d = _dist(mp_x, mp_y, tx, ty)
            if d < 0.5:
                continue
            est_ships = max(10, avail // 2)
            spd = _fleet_speed(est_ships)
            travel_t = d / spd
            if abs(travel_t - off) > 4:
                continue
            if not _sun_clear(mp_x, mp_y, tx, ty):
                continue
            ships_there = t[5] + off  # comet prod = 1
            needed = ships_there + 1
            if needed > avail:
                continue
            score = remaining_steps * 1.5 - needed * 0.8 - travel_t * 3.0
            if best is None or score > best[0]:
                best = (score, math.atan2(ty - mp_y, tx - mp_x), needed + 1, off)
            checked += 1
        return best

    def eval_planet(mp, t):
        mp_x, mp_y, avail = mp[2], mp[3], mp[5] - reserve(mp)
        if avail < 3:
            return None
        tx, ty = t[2], t[3]
        d = _dist(mp_x, mp_y, tx, ty)
        if d < 0.5:
            return None
        est_ships = max(10, avail // 2)
        spd = _fleet_speed(est_ships)
        est_time = d / spd

        # refine orbiting position
        tx, ty = _predict_planet_pos(t, initial_by_id, angular_velocity, step, est_time)
        d2 = _dist(mp_x, mp_y, tx, ty)
        est_time2 = d2 / spd
        tx, ty = _predict_planet_pos(t, initial_by_id, angular_velocity, step, est_time2)
        d_final = _dist(mp_x, mp_y, tx, ty)
        travel_t = d_final / spd

        if not _sun_clear(mp_x, mp_y, tx, ty):
            return None

        t_prod = t[6]
        ships_there = t[5] + int(travel_t * t_prod) + 1
        needed = ships_there
        if t[1] != -1:
            needed += 5 + int(travel_t * 0.5)
        if needed > avail:
            return None

        score = t_prod * remaining_steps - needed * 0.5 - travel_t * 3.0 - d_final * 0.15
        if t[1] == -1:
            score += 12
        return score, math.atan2(ty - mp_y, tx - mp_x), needed + 2

    moves = []
    my_planets_sorted = sorted(my_planets, key=lambda p: (-p[6], -p[5]))

    for mp in my_planets_sorted:
        avail = mp[5] - reserve(mp)
        if avail < 3:
            continue

        best_score = -1e12
        best_angle = 0.0
        best_ships = 0

        for t in planets:
            if t[1] == player:
                continue
            t_id = t[0]
            if t_id in comet_planet_ids:
                fut = comet_future.get(t_id)
                if not fut:
                    continue
                res = eval_comet(mp, t, fut)
                if res and res[0] > best_score:
                    best_score = res[0]
                    best_angle = res[1]
                    best_ships = res[2]
            else:
                res = eval_planet(mp, t)
                if res and res[0] > best_score:
                    best_score = res[0]
                    best_angle = res[1]
                    best_ships = res[2]

        if best_ships > 0:
            ships_to_send = int(min(avail, best_ships))
            if ships_to_send >= 3:
                moves.append([mp[0], best_angle, ships_to_send])

    return moves
