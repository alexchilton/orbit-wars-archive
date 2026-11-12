"""BullyBot - Python port for Orbit Wars.
Strategy: strongest planet (by ships) attacks weakest non-owned planet (fewest ships).
Only fires one fleet at a time (waits for it to land before firing again).
"""
import math

SUN_X, SUN_Y = 50.0, 50.0


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

    # Parse planets: [pid, owner, x, y, radius, ships, prod, ...]
    planets = []
    for p in planets_data:
        pid, owner, x, y, radius, ships, prod = p[:7]
        planets.append({'id': pid, 'owner': owner, 'x': x, 'y': y,
                        'ships': float(ships), 'prod': float(prod)})

    my_planets = [p for p in planets if p['owner'] == player]
    not_my = [p for p in planets if p['owner'] != player]

    if not my_planets or not not_my:
        return []

    # Only fire if no fleet of mine currently in flight
    my_fleets = [f for f in fleets_data if f[1] == player]
    if my_fleets:
        return []

    # Find my strongest planet (most ships)
    source = max(my_planets, key=lambda p: p['ships'])

    # Find weakest non-owned planet (fewest ships)
    dest = min(not_my, key=lambda p: p['ships'])

    num_ships = max(1, int(source['ships'] / 2))
    if num_ships < 1:
        return []

    angle = math.atan2(dest['y'] - source['y'], dest['x'] - source['x'])
    return [[source['id'], angle, num_ships]]
