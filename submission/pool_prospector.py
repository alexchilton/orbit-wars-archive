"""ProspectorBot - Python port for Orbit Wars.
Strategy: best ships/growth planet → best growth/ships non-owned planet (highest production ROI).
Only fires one fleet at a time.
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
    not_my = [p for p in planets if p['owner'] != player]

    if not my_planets or not not_my:
        return []

    my_fleets = [f for f in fleets_data if f[1] == player]
    if my_fleets:
        return []

    # Source: highest ships / (1 + growth) — planet that can afford to lose ships
    source = max(my_planets, key=lambda p: p['ships'] / (1.0 + p['prod']))

    # Dest: highest (1 + growth) / ships — best production-per-ship investment
    dest = max(not_my, key=lambda p: (1.0 + p['prod']) / max(p['ships'], 1.0))

    num_ships = max(1, int(source['ships'] / 2))
    if num_ships < 1:
        return []

    angle = math.atan2(dest['y'] - source['y'], dest['x'] - source['x'])
    return [[source['id'], angle, num_ships]]
