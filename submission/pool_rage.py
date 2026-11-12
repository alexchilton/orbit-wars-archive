"""RageBot - Python port for Orbit Wars.
Strategy: every planet with ships > 10 * production sends ALL ships to nearest enemy.
Extremely aggressive — ignores neutrals, just attacks enemies relentlessly.
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

    if not my_planets or not enemy_planets:
        return []

    moves = []
    for src in my_planets:
        # Only attack if ships > 10 * production rate
        if src['ships'] < 10.0 * max(src['prod'], 1.0):
            continue

        # Find nearest enemy planet
        best_enemy = min(
            enemy_planets,
            key=lambda e: math.hypot(e['x'] - src['x'], e['y'] - src['y'])
        )

        num_ships = max(1, int(src['ships']) - 1)
        angle = math.atan2(best_enemy['y'] - src['y'], best_enemy['x'] - src['x'])
        moves.append([src['id'], angle, num_ships])

    return moves
