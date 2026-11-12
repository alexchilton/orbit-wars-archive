#!/usr/bin/env python3
"""Test submitted vs fixed code on specific replay seeds."""
import importlib.util, os, sys, time, tarfile
CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

# Load SUBMITTED version (from tarball)
with tarfile.open(f'{CWD}/submission_v131_plus_curg_cl28.tar.gz', 'r:gz') as tar:
    main_py = tar.extractfile('main.py').read().decode('utf-8')
ns_sub = {'__name__': 'submitted', '__builtins__': __builtins__}
exec(compile(main_py, 'submitted_main.py', 'exec'), ns_sub)
submitted_agent = ns_sub['agent']

# Load FIXED version (current code)
def load_agent(path):
    spec = importlib.util.spec_from_file_location(f'_agent_{time.time()}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent
fixed_agent = load_agent(f'{CWD}/submission/main_v131_plus_2p.py')

# Load v131 original
v131_agent = load_agent('/Users/alexchilton/Downloads/main_v131.py')

import kaggle_environments

SEEDS = {
    '75940524': 1984508306,
    '75939703': 839397798,
    '75939481': 143877781,
    '75939243': 1175431795,
    '75938658': 219091541,
    '75939022': 2117651738,
}

for replay_id, seed in SEEDS.items():
    print(f"\n{'='*50}")
    print(f"Replay {replay_id}, seed {seed}")
    print(f"{'='*50}")
    
    for version_name, agent in [('SUBMITTED', submitted_agent), ('FIXED', fixed_agent)]:
        env = kaggle_environments.make('orbit_wars', debug=True, configuration={'seed': seed})
        env.run([agent, v131_agent])
        
        trap_turns = 0
        for step_idx in range(min(60, len(env.steps))):
            step_data = env.steps[step_idx]
            p0 = step_data[0].get('action', None)
            p1 = step_data[1].get('action', None)
            p0_empty = p0 is not None and (p0 == [] or p0 == [[]])
            p1_empty = p1 is not None and (p1 == [] or p1 == [[]])
            if p0_empty and not p1_empty:
                trap_turns += 1
        
        final = [env.state[j]['reward'] for j in range(2)]
        result = 'WIN' if final[0] > final[1] else 'LOSS'
        print(f"  {version_name:10s}: traps={trap_turns:2d}  result={result}  ({final[0]:.0f} vs {final[1]:.0f})")
