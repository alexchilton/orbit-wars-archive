"""DualBot - Python port for Orbit Wars.
Strategy: switches between attack mode and expand mode based on production & ship advantage.
- If winning ships AND production: attack mode (target only enemies, 1 fleet cap)
- If winning ships but not production: expand mode (3 fleet cap, any non-owned)
- If not winning ships but winning prod: conservative (1 fleet cap, any non-owned)
- If losing both: aggressive expand (5 fleet cap, best prod targets)
"""
import math


def _parse(obs):
    if isinstance(obs, dict):
        player = obs.get('player', 0)
        planets_data = obs.get('planets', [])
        fleets_data = obs.get('fleets', [])
    else:
        player = getattr(obs, 'player', 0)
        planets_data = getattr(obs, 'planets', [])
        fleets_data = getattr(obs, 'fleets', [])
    return player, planets_data, fleets_data


def agent(obs):
    player, planets_data, fleets_data = _parse(obs)

    planets = []
    for p in planets_data:
        pid, owner, x, y, radius, ships, prod = p[:7]
        planets.append({'id': pid, 'owner': owner, 'x': x, 'y': y,
                        'ships': float(ships), 'prod': float(prod)})

    my_planets = [p for p in planets if p['owner'] == player]
    enemy_planets = [p for p in planets if p['owner'] != player and p['owner'] != -1]
    not_my = [p for p in planets if p['owner'] != player]

    if not my_planets or not not_my:
        return []

    my_ships = sum(p['ships'] for p in my_planets)
    my_prod = sum(p['prod'] for p in my_planets)
    enemy_ships = sum(p['ships'] for p in enemy_planets)
    enemy_prod = sum(p['prod'] for p in enemy_planets)

    winning_ships = my_ships > enemy_ships
    winning_prod = my_prod > enemy_prod

    # Determine mode
    if winning_ships and winning_prod:
        fleet_cap = 1
        attack_mode = True
        candidates = enemy_planets if enemy_planets else not_my
    elif winning_ships and not winning_prod:
        fleet_cap = 3
        attack_mode = False
        candidates = not_my
    elif not winning_ships and winning_prod:
        fleet_cap = 1
        attack_mode = False
        candidates = not_my
    else:
        fleet_cap = 5
        attack_mode = False
        candidates = not_my

    if not candidates:
        return []

    my_fleets = [f for f in fleets_data if f[1] == player]
    if len(my_fleets) >= fleet_cap:
        return []

    # Source: highest ships / (1 + growth)
    source = max(my_planets, key=lambda p: p['ships'] / (1.0 + p['prod']))

    # Dest: in attack mode, weakest enemy; otherwise best prod/ships ratio
    if attack_mode:
        dest = min(candidates, key=lambda p: p['ships'])
    else:
        dest = max(candidates, key=lambda p: (1.0 + p['prod']) / max(p['ships'], 1.0))

    num_ships = max(1, int(source['ships'] / 2))
    if num_ships < 1:
        return []

    angle = math.atan2(dest['y'] - source['y'], dest['x'] - source['x'])
    return [[source['id'], angle, num_ships]]
