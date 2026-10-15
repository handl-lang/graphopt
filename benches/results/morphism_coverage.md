# Morphism-layer coverage on real checkpoints

How much of a real transformer the morphism engine can see: block-lift rates, per-law match/fire/graft ledgers on llama2.c stories15M/stories110M, guarded default-stack outcome, delivered-module verify + latency vs eager.

- **when**: 2026-10-14T02:30:01+00:00 (git `c1a9521` clean)
- **device**: `cpu` — 12th Gen Intel(R) Core(TM) i5-12500H
- **torch**: 2.14.0+cu130 · **cuda**: 13.0 · **python**: 3.13.9
- **config**: `checkpoints`={'stories15M': '/home/luis/.cache/catopt/stories15M.bin', 'stories110M': '/home/luis/.cache/catopt/stories110M.bin'} · `coverage`={'stories15M': {'model': 'stories15M', 'ckpt': '/home/luis/.cache/catopt/stories15M.bin', 'cfg': {'dim': 288, 'hidden': 768, 'n_layers': 6, 'n_heads': 6, 'vocab': 32000, 'seq_len': 256}, 'seq': 64, 'load_s': 11.0, 'coverage': {'lift': {'n_nodes': 8, 'n_lifted': 8, 'n_opaque': 0, 'input_kinds': {'activation': 8, 'const_table': 12}, 'input_roles': {'stream': 8, 'table': 12}, 'wire_kinds': {'opaque': 3, 'chain': 4}, 'wires': [['emb', 'blocks.0', 'opaque'], ['blocks.0', 'blocks.1', 'chain'], ['blocks.1', 'blocks.2', 'chain'], ['blocks.2', 'blocks.3', 'chain'], ['blocks.3', 'blocks.4', 'chain'], ['blocks.4', 'blocks.5', 'opaque'], ['blocks.5', 'head', 'opaque']], 'blocks': {'emb': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': [], 'out_proj': [], 'norm': 'none', 'norm_affine': False, 'norm_pre': False, 'act': [], 'residual': False, 'shape': [[1, 64], [1, 64, 288]], 'inputs': [{'index': 0, 'name': 'input', 'kind': 'activation', 'role': 'stream'}]}}, 'blocks.0': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.1': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.2': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.3': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.4': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.5': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'head': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_weight'], 'out_proj': ['p_weight'], 'norm': 'none', 'norm_affine': False, 'norm_pre': False, 'act': [], 'residual': False, 'shape': [[64, 288], [64, 32000]], 'inputs': [{'index': 0, 'name': 'input', 'kind': 'activation', 'role': 'stream'}]}}}}, 'candidacy': {'window_compose': {'matches': 1, 'sites': [['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4']], 'boundaries': ['chain+chain+chain+chain'], 'details': ['blocks.0->…->blocks.4: 5-block out∘in window']}, 'residual_reassoc': {'matches': 0, 'sites': [], 'boundaries': [], 'details': []}, 'out_in_compose': {'matches': 4, 'sites': [['blocks.0', 'blocks.1'], ['blocks.1', 'blocks.2'], ['blocks.2', 'blocks.3'], ['blocks.3', 'blocks.4']], 'boundaries': ['chain', 'chain', 'chain', 'chain'], 'details': ['blocks.0.out_proj ∘ blocks.1.in_projs', 'blocks.1.out_proj ∘ blocks.2.in_projs', 'blocks.2.out_proj ∘ blocks.3.in_projs', 'blocks.3.out_proj ∘ blocks.4.in_projs']}, 'residual_absorb': {'matches': 0, 'sites': [], 'boundaries': [], 'details': []}, 'norm_cascade': {'matches': 6, 'sites': [['blocks.0'], ['blocks.1'], ['blocks.2'], ['blocks.3'], ['blocks.4'], ['blocks.5']], 'boundaries': ['intra', 'intra', 'intra', 'intra', 'intra', 'intra'], 'details': ['blocks.0 pre-norm gain folds into in_projs', 'blocks.1 pre-norm gain folds into in_projs', 'blocks.2 pre-norm gain folds into in_projs', 'blocks.3 pre-norm gain folds into in_projs', 'blocks.4 pre-norm gain folds into in_projs', 'blocks.5 pre-norm gain folds into in_projs']}, 'weight_tie': {'matches': 22, 'sites': [['blocks.0'], ['blocks.1'], ['blocks.2'], ['blocks.3'], ['blocks.4'], ['blocks.5'], ['emb', 'head'], ['blocks.0', 'blocks.1'], ['blocks.0', 'blocks.2'], ['blocks.0', 'blocks.3'], ['blocks.0', 'blocks.4'], ['blocks.0', 'blocks.5'], ['blocks.1', 'blocks.2'], ['blocks.1', 'blocks.3'], ['blocks.1', 'blocks.4'], ['blocks.1', 'blocks.5'], ['blocks.2', 'blocks.3'], ['blocks.2', 'blocks.4'], ['blocks.2', 'blocks.5'], ['blocks.3', 'blocks.4'], ['blocks.3', 'blocks.5'], ['blocks.4', 'blocks.5']], 'boundaries': ['tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie'], 'details': ['blocks.0: same-shape weights', 'blocks.1: same-shape weights', 'blocks.2: same-shape weights', 'blocks.3: same-shape weights', 'blocks.4: same-shape weights', 'blocks.5: same-shape weights', 'emb~head: tied weights', 'blocks.0~blocks.1: tied weights', 'blocks.0~blocks.2: tied weights', 'blocks.0~blocks.3: tied weights', 'blocks.0~blocks.4: tied weights', 'blocks.0~blocks.5: tied weights', 'blocks.1~blocks.2: tied weights', 'blocks.1~blocks.3: tied weights', 'blocks.1~blocks.4: tied weights', 'blocks.1~blocks.5: tied weights', 'blocks.2~blocks.3: tied weights', 'blocks.2~blocks.4: tied weights', 'blocks.2~blocks.5: tied weights', 'blocks.3~blocks.4: tied weights', 'blocks.3~blocks.5: tied weights', 'blocks.4~blocks.5: tied weights']}, 'kv_latent_share': {'matches': 6, 'sites': [['blocks.0'], ['blocks.1'], ['blocks.2'], ['blocks.3'], ['blocks.4'], ['blocks.5']], 'boundaries': ['intra', 'intra', 'intra', 'intra', 'intra', 'intra'], 'details': ['blocks.0: shared-data kv projections', 'blocks.1: shared-data kv projections', 'blocks.2: shared-data kv projections', 'blocks.3: shared-data kv projections', 'blocks.4: shared-data kv projections', 'blocks.5: shared-data kv projections']}}, 'seconds': 7.63}, 'gaps': {'tied_param_pairs': [{'a': 'emb.weight', 'b': 'head.weight', 'shape': [32000, 288]}], 'norm_ops_vs_sig': {'emb': {'norm_ops': {}, 'sig_norm': 'none'}, 'blocks.0': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.1': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.2': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.3': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.4': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.5': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'head': {'norm_ops': {}, 'sig_norm': 'none'}}, 'opaque_wires': [['emb', 'blocks.0', 'opaque'], ['blocks.4', 'blocks.5', 'opaque'], ['blocks.5', 'head', 'opaque']]}, 'default_stack': {'laws': 'all', 'optimize_rest': True, 'exit': 0, 'seconds': 224.8, 'died_at': [], 'progress_tail': [['DONE', '', '']], 'stderr_tail': '', 'stats': {'morphism': True, 'n_blocks': 8, 'n_lifted': 8, 'sigs': {'emb': {'in_projs': [], 'out_proj': [], 'tables': ['p_weight'], 'norm': 'none', 'norm_affine': False, 'norm_pre': False, 'act': [], 'residual': False, 'shape': [[1, 64], [1, 64, 288]], 'inputs': [{'index': 0, 'name': 'input', 'kind': 'activation', 'role': 'stream'}]}, 'blocks.0': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.1': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.2': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.3': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.4': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.5': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 288], [64, 288]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'head': {'in_projs': ['p_weight'], 'out_proj': ['p_weight'], 'tables': [], 'norm': 'none', 'norm_affine': False, 'norm_pre': False, 'act': [], 'residual': False, 'shape': [[64, 288], [64, 32000]], 'inputs': [{'index': 0, 'name': 'input', 'kind': 'activation', 'role': 'stream'}]}}, 'wires': [['emb', 'blocks.0', 'opaque'], ['blocks.0', 'blocks.1', 'chain'], ['blocks.1', 'blocks.2', 'chain'], ['blocks.2', 'blocks.3', 'chain'], ['blocks.3', 'blocks.4', 'chain'], ['blocks.4', 'blocks.5', 'opaque'], ['blocks.5', 'head', 'opaque']], 'matches': {'window_compose:blocks.0+blocks.1+blocks.2+blocks.3+blocks.4': {'boundary': 'chain+chain+chain+chain', 'detail': 'blocks.0->…->blocks.4: 5-block out∘in window', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 663432192.0, 'cost_after': 663432192.0}, 'out_in_compose:blocks.0+blocks.1': {'boundary': 'chain', 'detail': 'blocks.0.out_proj ∘ blocks.1.in_projs', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'out_in_compose:blocks.1+blocks.2': {'boundary': 'chain', 'detail': 'blocks.1.out_proj ∘ blocks.2.in_projs', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'out_in_compose:blocks.2+blocks.3': {'boundary': 'chain', 'detail': 'blocks.2.out_proj ∘ blocks.3.in_projs', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'out_in_compose:blocks.3+blocks.4': {'boundary': 'chain', 'detail': 'blocks.3.out_proj ∘ blocks.4.in_projs', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'norm_cascade:blocks.0': {'boundary': 'intra', 'detail': 'blocks.0 pre-norm gain folds into in_projs', 'status': 'grafted', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.6346184484560376e-07}, 'norm_cascade:blocks.1': {'boundary': 'intra', 'detail': 'blocks.1 pre-norm gain folds into in_projs', 'status': 'grafted', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.647190394666863e-07}, 'norm_cascade:blocks.2': {'boundary': 'intra', 'detail': 'blocks.2 pre-norm gain folds into in_projs', 'status': 'grafted', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.0895082609941918e-07}, 'norm_cascade:blocks.3': {'boundary': 'intra', 'detail': 'blocks.3 pre-norm gain folds into in_projs', 'status': 'grafted', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 1.0293398914506104e-07}, 'norm_cascade:blocks.4': {'boundary': 'intra', 'detail': 'blocks.4 pre-norm gain folds into in_projs', 'status': 'grafted', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 1.0775967726300525e-07}, 'norm_cascade:blocks.5': {'boundary': 'intra', 'detail': 'blocks.5 pre-norm gain folds into in_projs', 'status': 'grafted', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.050507441414449e-07}, 'weight_tie:blocks.0': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:emb+head': {'boundary': 'tie', 'detail': 'emb~head: tied weights', 'status': 'grafted', 'tied': [['emb__p_weight', 'head__p_weight']], 'rel_diff': 0.0}, 'weight_tie:blocks.0+blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.0': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}}, 'morphism_fires': {'norm_cascade': 6, 'weight_tie': 1}, 'n_rewritten': 8, 'blocks': {'emb': {'status': 'rewritten', 'law': 'weight_tie'}, 'blocks.0': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.1': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.2': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.3': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.4': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.5': {'status': 'rewritten', 'law': 'norm_cascade'}, 'head': {'status': 'rewritten', 'law': 'weight_tie'}}, 'in_place': False, 'shared_params': True, 'end_to_end': {'max_abs_diff': 2.47955322265625e-05, 'max_rel_diff': 1.7041540118787826e-06}, 'wall_time_s': 186.33752822875977}, 'outcome': 'completed'}, 'delivery': {'laws': ['residual_reassoc', 'out_in_compose', 'residual_absorb', 'norm_cascade', 'weight_tie', 'kv_latent_share'], 'optimize_rest': True, 'window_excluded': True, 'window_exclusion_reason': 'window match declined under the guarded run', 'seconds': 3.3, 'n_rewritten': 8, 'morphism_fires': {'norm_cascade': 6, 'weight_tie': 1}, 'end_to_end': {'max_abs_diff': 2.47955322265625e-05, 'max_rel_diff': 1.7041540118787826e-06}, 'in_place': False, 'per_law': {'out_in_compose': {'matched': 4, 'grafted': 0, 'declined': 4, 'skipped': 0, 'reasons': {'no_improvement': 4}}, 'norm_cascade': {'matched': 6, 'grafted': 6, 'declined': 0, 'skipped': 0, 'reasons': {}}, 'weight_tie': {'matched': 22, 'grafted': 1, 'declined': 0, 'skipped': 21, 'reasons': {'node already rewritten': 21}}, 'kv_latent_share': {'matched': 6, 'grafted': 0, 'declined': 0, 'skipped': 6, 'reasons': {'node already rewritten': 6}}}, 'matches': {'out_in_compose:blocks.0+blocks.1': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.0.out_proj ∘ blocks.1.in_projs', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'out_in_compose:blocks.1+blocks.2': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.1.out_proj ∘ blocks.2.in_projs', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'out_in_compose:blocks.2+blocks.3': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.2.out_proj ∘ blocks.3.in_projs', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'out_in_compose:blocks.3+blocks.4': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.3.out_proj ∘ blocks.4.in_projs', 'cost_before': 265374720.0, 'cost_after': 265374720.0}, 'norm_cascade:blocks.0': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.0 pre-norm gain folds into in_projs', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.6346184484560376e-07}, 'norm_cascade:blocks.1': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.1 pre-norm gain folds into in_projs', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.647190394666863e-07}, 'norm_cascade:blocks.2': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.2 pre-norm gain folds into in_projs', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.0895082609941918e-07}, 'norm_cascade:blocks.3': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.3 pre-norm gain folds into in_projs', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 1.0293398914506104e-07}, 'norm_cascade:blocks.4': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.4 pre-norm gain folds into in_projs', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 1.0775967726300525e-07}, 'norm_cascade:blocks.5': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.5 pre-norm gain folds into in_projs', 'cost_before': 132688896.0, 'cost_after': 132652032.0, 'rel_diff': 2.050507441414449e-07}, 'weight_tie:blocks.0': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:emb+head': {'status': 'grafted', 'boundary': 'tie', 'detail': 'emb~head: tied weights', 'rel_diff': 0.0, 'tied': [['emb__p_weight', 'head__p_weight']]}, 'weight_tie:blocks.0+blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.0': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}}, 'blocks': {'emb': {'status': 'rewritten', 'law': 'weight_tie'}, 'blocks.0': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.1': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.2': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.3': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.4': {'status': 'rewritten', 'law': 'norm_cascade'}, 'blocks.5': {'status': 'rewritten', 'law': 'norm_cascade'}, 'head': {'status': 'rewritten', 'law': 'weight_tie'}}}, 'verify_fresh': {'passed': True, 'max_rel': 1.4674530641600703e-06}, 'timing': {'eager': 42.035, 'morphism': 34.177, 'speedup': 1.2299}}, 'stories110M': {'model': 'stories110M', 'ckpt': '/home/luis/.cache/catopt/stories110M.bin', 'cfg': {'dim': 768, 'hidden': 2048, 'n_layers': 12, 'n_heads': 12, 'vocab': 32000, 'seq_len': 1024}, 'seq': 64, 'load_s': 1.5, 'coverage': {'lift': {'n_nodes': 14, 'n_lifted': 14, 'n_opaque': 0, 'input_kinds': {'activation': 14, 'const_table': 24}, 'input_roles': {'stream': 14, 'table': 24}, 'wire_kinds': {'opaque': 3, 'chain': 10}, 'wires': [['emb', 'blocks.0', 'opaque'], ['blocks.0', 'blocks.1', 'chain'], ['blocks.1', 'blocks.2', 'chain'], ['blocks.2', 'blocks.3', 'chain'], ['blocks.3', 'blocks.4', 'chain'], ['blocks.4', 'blocks.5', 'chain'], ['blocks.5', 'blocks.6', 'chain'], ['blocks.6', 'blocks.7', 'chain'], ['blocks.7', 'blocks.8', 'chain'], ['blocks.8', 'blocks.9', 'chain'], ['blocks.9', 'blocks.10', 'chain'], ['blocks.10', 'blocks.11', 'opaque'], ['blocks.11', 'head', 'opaque']], 'blocks': {'emb': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': [], 'out_proj': [], 'norm': 'none', 'norm_affine': False, 'norm_pre': False, 'act': [], 'residual': False, 'shape': [[1, 64], [1, 64, 768]], 'inputs': [{'index': 0, 'name': 'input', 'kind': 'activation', 'role': 'stream'}]}}, 'blocks.0': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.1': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.2': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.3': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.4': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.5': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.6': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.7': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.8': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.9': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.10': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'blocks.11': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}}, 'head': {'opaque': False, 'note': None, 'calls': 1, 'sig': {'in_projs': ['p_weight'], 'out_proj': ['p_weight'], 'norm': 'none', 'norm_affine': False, 'norm_pre': False, 'act': [], 'residual': False, 'shape': [[64, 768], [64, 32000]], 'inputs': [{'index': 0, 'name': 'input', 'kind': 'activation', 'role': 'stream'}]}}}}, 'candidacy': {'window_compose': {'matches': 1, 'sites': [['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10']], 'boundaries': ['chain+chain+chain+chain+chain+chain+chain+chain+chain+chain'], 'details': ['blocks.0->…->blocks.10: 11-block out∘in window']}, 'residual_reassoc': {'matches': 0, 'sites': [], 'boundaries': [], 'details': []}, 'out_in_compose': {'matches': 10, 'sites': [['blocks.0', 'blocks.1'], ['blocks.1', 'blocks.2'], ['blocks.2', 'blocks.3'], ['blocks.3', 'blocks.4'], ['blocks.4', 'blocks.5'], ['blocks.5', 'blocks.6'], ['blocks.6', 'blocks.7'], ['blocks.7', 'blocks.8'], ['blocks.8', 'blocks.9'], ['blocks.9', 'blocks.10']], 'boundaries': ['chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain'], 'details': ['blocks.0.out_proj ∘ blocks.1.in_projs', 'blocks.1.out_proj ∘ blocks.2.in_projs', 'blocks.2.out_proj ∘ blocks.3.in_projs', 'blocks.3.out_proj ∘ blocks.4.in_projs', 'blocks.4.out_proj ∘ blocks.5.in_projs', 'blocks.5.out_proj ∘ blocks.6.in_projs', 'blocks.6.out_proj ∘ blocks.7.in_projs', 'blocks.7.out_proj ∘ blocks.8.in_projs', 'blocks.8.out_proj ∘ blocks.9.in_projs', 'blocks.9.out_proj ∘ blocks.10.in_projs']}, 'residual_absorb': {'matches': 0, 'sites': [], 'boundaries': [], 'details': []}, 'norm_cascade': {'matches': 12, 'sites': [['blocks.0'], ['blocks.1'], ['blocks.2'], ['blocks.3'], ['blocks.4'], ['blocks.5'], ['blocks.6'], ['blocks.7'], ['blocks.8'], ['blocks.9'], ['blocks.10'], ['blocks.11']], 'boundaries': ['intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra'], 'details': ['blocks.0 pre-norm gain folds into in_projs', 'blocks.1 pre-norm gain folds into in_projs', 'blocks.2 pre-norm gain folds into in_projs', 'blocks.3 pre-norm gain folds into in_projs', 'blocks.4 pre-norm gain folds into in_projs', 'blocks.5 pre-norm gain folds into in_projs', 'blocks.6 pre-norm gain folds into in_projs', 'blocks.7 pre-norm gain folds into in_projs', 'blocks.8 pre-norm gain folds into in_projs', 'blocks.9 pre-norm gain folds into in_projs', 'blocks.10 pre-norm gain folds into in_projs', 'blocks.11 pre-norm gain folds into in_projs']}, 'weight_tie': {'matches': 79, 'sites': [['blocks.0'], ['blocks.1'], ['blocks.2'], ['blocks.3'], ['blocks.4'], ['blocks.5'], ['blocks.6'], ['blocks.7'], ['blocks.8'], ['blocks.9'], ['blocks.10'], ['blocks.11'], ['emb', 'head'], ['blocks.0', 'blocks.1'], ['blocks.0', 'blocks.2'], ['blocks.0', 'blocks.3'], ['blocks.0', 'blocks.4'], ['blocks.0', 'blocks.5'], ['blocks.0', 'blocks.6'], ['blocks.0', 'blocks.7'], ['blocks.0', 'blocks.8'], ['blocks.0', 'blocks.9'], ['blocks.0', 'blocks.10'], ['blocks.0', 'blocks.11'], ['blocks.1', 'blocks.2'], ['blocks.1', 'blocks.3'], ['blocks.1', 'blocks.4'], ['blocks.1', 'blocks.5'], ['blocks.1', 'blocks.6'], ['blocks.1', 'blocks.7'], ['blocks.1', 'blocks.8'], ['blocks.1', 'blocks.9'], ['blocks.1', 'blocks.10'], ['blocks.1', 'blocks.11'], ['blocks.2', 'blocks.3'], ['blocks.2', 'blocks.4'], ['blocks.2', 'blocks.5'], ['blocks.2', 'blocks.6'], ['blocks.2', 'blocks.7'], ['blocks.2', 'blocks.8'], ['blocks.2', 'blocks.9'], ['blocks.2', 'blocks.10'], ['blocks.2', 'blocks.11'], ['blocks.3', 'blocks.4'], ['blocks.3', 'blocks.5'], ['blocks.3', 'blocks.6'], ['blocks.3', 'blocks.7'], ['blocks.3', 'blocks.8'], ['blocks.3', 'blocks.9'], ['blocks.3', 'blocks.10'], ['blocks.3', 'blocks.11'], ['blocks.4', 'blocks.5'], ['blocks.4', 'blocks.6'], ['blocks.4', 'blocks.7'], ['blocks.4', 'blocks.8'], ['blocks.4', 'blocks.9'], ['blocks.4', 'blocks.10'], ['blocks.4', 'blocks.11'], ['blocks.5', 'blocks.6'], ['blocks.5', 'blocks.7'], ['blocks.5', 'blocks.8'], ['blocks.5', 'blocks.9'], ['blocks.5', 'blocks.10'], ['blocks.5', 'blocks.11'], ['blocks.6', 'blocks.7'], ['blocks.6', 'blocks.8'], ['blocks.6', 'blocks.9'], ['blocks.6', 'blocks.10'], ['blocks.6', 'blocks.11'], ['blocks.7', 'blocks.8'], ['blocks.7', 'blocks.9'], ['blocks.7', 'blocks.10'], ['blocks.7', 'blocks.11'], ['blocks.8', 'blocks.9'], ['blocks.8', 'blocks.10'], ['blocks.8', 'blocks.11'], ['blocks.9', 'blocks.10'], ['blocks.9', 'blocks.11'], ['blocks.10', 'blocks.11']], 'boundaries': ['tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie', 'tie'], 'details': ['blocks.0: same-shape weights', 'blocks.1: same-shape weights', 'blocks.2: same-shape weights', 'blocks.3: same-shape weights', 'blocks.4: same-shape weights', 'blocks.5: same-shape weights', 'blocks.6: same-shape weights', 'blocks.7: same-shape weights', 'blocks.8: same-shape weights', 'blocks.9: same-shape weights', 'blocks.10: same-shape weights', 'blocks.11: same-shape weights', 'emb~head: tied weights', 'blocks.0~blocks.1: tied weights', 'blocks.0~blocks.2: tied weights', 'blocks.0~blocks.3: tied weights', 'blocks.0~blocks.4: tied weights', 'blocks.0~blocks.5: tied weights', 'blocks.0~blocks.6: tied weights', 'blocks.0~blocks.7: tied weights', 'blocks.0~blocks.8: tied weights', 'blocks.0~blocks.9: tied weights', 'blocks.0~blocks.10: tied weights', 'blocks.0~blocks.11: tied weights', 'blocks.1~blocks.2: tied weights', 'blocks.1~blocks.3: tied weights', 'blocks.1~blocks.4: tied weights', 'blocks.1~blocks.5: tied weights', 'blocks.1~blocks.6: tied weights', 'blocks.1~blocks.7: tied weights', 'blocks.1~blocks.8: tied weights', 'blocks.1~blocks.9: tied weights', 'blocks.1~blocks.10: tied weights', 'blocks.1~blocks.11: tied weights', 'blocks.2~blocks.3: tied weights', 'blocks.2~blocks.4: tied weights', 'blocks.2~blocks.5: tied weights', 'blocks.2~blocks.6: tied weights', 'blocks.2~blocks.7: tied weights', 'blocks.2~blocks.8: tied weights', 'blocks.2~blocks.9: tied weights', 'blocks.2~blocks.10: tied weights', 'blocks.2~blocks.11: tied weights', 'blocks.3~blocks.4: tied weights', 'blocks.3~blocks.5: tied weights', 'blocks.3~blocks.6: tied weights', 'blocks.3~blocks.7: tied weights', 'blocks.3~blocks.8: tied weights', 'blocks.3~blocks.9: tied weights', 'blocks.3~blocks.10: tied weights', 'blocks.3~blocks.11: tied weights', 'blocks.4~blocks.5: tied weights', 'blocks.4~blocks.6: tied weights', 'blocks.4~blocks.7: tied weights', 'blocks.4~blocks.8: tied weights', 'blocks.4~blocks.9: tied weights', 'blocks.4~blocks.10: tied weights', 'blocks.4~blocks.11: tied weights', 'blocks.5~blocks.6: tied weights', 'blocks.5~blocks.7: tied weights', 'blocks.5~blocks.8: tied weights', 'blocks.5~blocks.9: tied weights', 'blocks.5~blocks.10: tied weights', 'blocks.5~blocks.11: tied weights', 'blocks.6~blocks.7: tied weights', 'blocks.6~blocks.8: tied weights', 'blocks.6~blocks.9: tied weights', 'blocks.6~blocks.10: tied weights', 'blocks.6~blocks.11: tied weights', 'blocks.7~blocks.8: tied weights', 'blocks.7~blocks.9: tied weights', 'blocks.7~blocks.10: tied weights', 'blocks.7~blocks.11: tied weights', 'blocks.8~blocks.9: tied weights', 'blocks.8~blocks.10: tied weights', 'blocks.8~blocks.11: tied weights', 'blocks.9~blocks.10: tied weights', 'blocks.9~blocks.11: tied weights', 'blocks.10~blocks.11: tied weights']}, 'kv_latent_share': {'matches': 12, 'sites': [['blocks.0'], ['blocks.1'], ['blocks.10'], ['blocks.11'], ['blocks.2'], ['blocks.3'], ['blocks.4'], ['blocks.5'], ['blocks.6'], ['blocks.7'], ['blocks.8'], ['blocks.9']], 'boundaries': ['intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra', 'intra'], 'details': ['blocks.0: shared-data kv projections', 'blocks.1: shared-data kv projections', 'blocks.10: shared-data kv projections', 'blocks.11: shared-data kv projections', 'blocks.2: shared-data kv projections', 'blocks.3: shared-data kv projections', 'blocks.4: shared-data kv projections', 'blocks.5: shared-data kv projections', 'blocks.6: shared-data kv projections', 'blocks.7: shared-data kv projections', 'blocks.8: shared-data kv projections', 'blocks.9: shared-data kv projections']}}, 'seconds': 1.83}, 'gaps': {'tied_param_pairs': [{'a': 'emb.weight', 'b': 'head.weight', 'shape': [32000, 768]}], 'norm_ops_vs_sig': {'emb': {'norm_ops': {}, 'sig_norm': 'none'}, 'blocks.0': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.1': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.2': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.3': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.4': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.5': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.6': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.7': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.8': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.9': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.10': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'blocks.11': {'norm_ops': {'rms_norm': 2}, 'sig_norm': 'rms'}, 'head': {'norm_ops': {}, 'sig_norm': 'none'}}, 'opaque_wires': [['emb', 'blocks.0', 'opaque'], ['blocks.10', 'blocks.11', 'opaque'], ['blocks.11', 'head', 'opaque']]}, 'default_stack': {'laws': 'all', 'optimize_rest': False, 'exit': 0, 'seconds': 46.7, 'died_at': [], 'progress_tail': [['DONE', '', '']], 'stderr_tail': '', 'stats': {'morphism': True, 'n_blocks': 14, 'n_lifted': 5, 'sigs': {'emb': {'in_projs': [], 'out_proj': [], 'tables': ['p_weight'], 'norm': 'none', 'norm_affine': False, 'norm_pre': False, 'act': [], 'residual': False, 'shape': [[1, 64], [1, 64, 768]], 'inputs': [{'index': 0, 'name': 'input', 'kind': 'activation', 'role': 'stream'}]}, 'blocks.0': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.1': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.2': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.3': {'in_projs': ['p_wv_weight', 'p_wk_weight', 'p_wq_weight'], 'out_proj': ['p_w2_weight'], 'tables': [], 'norm': 'rms', 'norm_affine': True, 'norm_pre': True, 'act': ['sdpa', 'silu'], 'residual': True, 'shape': [[64, 768], [64, 768]], 'inputs': [{'index': 0, 'name': 'h', 'kind': 'activation', 'role': 'stream'}, {'index': 1, 'name': 'cos', 'kind': 'const_table', 'role': 'table'}, {'index': 2, 'name': 'sin', 'kind': 'const_table', 'role': 'table'}]}, 'blocks.4': None, 'blocks.5': None, 'blocks.6': None, 'blocks.7': None, 'blocks.8': None, 'blocks.9': None, 'blocks.10': None, 'blocks.11': None, 'head': None}, 'wires': [['emb', 'blocks.0', 'opaque'], ['blocks.0', 'blocks.1', 'chain'], ['blocks.1', 'blocks.2', 'chain'], ['blocks.2', 'blocks.3', 'chain'], ['blocks.3', 'blocks.4', 'chain'], ['blocks.4', 'blocks.5', 'opaque'], ['blocks.5', 'blocks.6', 'opaque'], ['blocks.6', 'blocks.7', 'opaque'], ['blocks.7', 'blocks.8', 'opaque'], ['blocks.8', 'blocks.9', 'opaque'], ['blocks.9', 'blocks.10', 'opaque'], ['blocks.10', 'blocks.11', 'opaque'], ['blocks.11', 'head', 'opaque']], 'matches': {'window_compose:blocks.0+blocks.1+blocks.2+blocks.3': {'boundary': 'chain+chain+chain', 'detail': 'blocks.0->…->blocks.3: 4-block out∘in window', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 3680243712.0, 'cost_after': 3680243712.0}, 'out_in_compose:blocks.0+blocks.1': {'boundary': 'chain', 'detail': 'blocks.0.out_proj ∘ blocks.1.in_projs', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.1+blocks.2': {'boundary': 'chain', 'detail': 'blocks.1.out_proj ∘ blocks.2.in_projs', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.2+blocks.3': {'boundary': 'chain', 'detail': 'blocks.2.out_proj ∘ blocks.3.in_projs', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'norm_cascade:blocks.0': {'status': 'declined', 'reason': 'error', 'error': "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"}, 'norm_cascade:blocks.1': {'status': 'declined', 'reason': 'error', 'error': "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"}, 'norm_cascade:blocks.2': {'status': 'declined', 'reason': 'error', 'error': "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"}, 'norm_cascade:blocks.3': {'status': 'declined', 'reason': 'error', 'error': "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"}, 'weight_tie:blocks.0': {'boundary': 'tie', 'detail': 'blocks.0: same-shape weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.1': {'boundary': 'tie', 'detail': 'blocks.1: same-shape weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.2': {'boundary': 'tie', 'detail': 'blocks.2: same-shape weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.3': {'boundary': 'tie', 'detail': 'blocks.3: same-shape weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.0+blocks.1': {'boundary': 'tie', 'detail': 'blocks.0~blocks.1: tied weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.0+blocks.2': {'boundary': 'tie', 'detail': 'blocks.0~blocks.2: tied weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.0+blocks.3': {'boundary': 'tie', 'detail': 'blocks.0~blocks.3: tied weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.1+blocks.2': {'boundary': 'tie', 'detail': 'blocks.1~blocks.2: tied weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.1+blocks.3': {'boundary': 'tie', 'detail': 'blocks.1~blocks.3: tied weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'weight_tie:blocks.2+blocks.3': {'boundary': 'tie', 'detail': 'blocks.2~blocks.3: tied weights', 'status': 'declined', 'reason': 'no_tied_values'}, 'kv_latent_share:blocks.0': {'boundary': 'intra', 'detail': 'blocks.0: shared-data kv projections', 'status': 'declined', 'reason': 'no certified common factor'}, 'kv_latent_share:blocks.1': {'status': 'declined', 'reason': 'error', 'error': "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 2359296 bytes. Error code 12 (Cannot allocate memory)"}, 'kv_latent_share:blocks.2': {'boundary': 'intra', 'detail': 'blocks.2: shared-data kv projections', 'status': 'declined', 'reason': 'no certified common factor'}, 'kv_latent_share:blocks.3': {'boundary': 'intra', 'detail': 'blocks.3: shared-data kv projections', 'status': 'declined', 'reason': 'no certified common factor'}}, 'morphism_fires': {}, 'n_rewritten': 0, 'blocks': {}, 'in_place': False, 'shared_params': True, 'end_to_end': {'error': "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 8192000 bytes. Error code 12 (Cannot allocate memory)"}, 'wall_time_s': 43.53308725357056}, 'outcome': 'completed'}, 'delivery': {'laws': ['residual_reassoc', 'out_in_compose', 'residual_absorb', 'norm_cascade', 'weight_tie', 'kv_latent_share'], 'optimize_rest': False, 'window_excluded': True, 'window_exclusion_reason': 'window match declined under the guarded run', 'seconds': 4.9, 'n_rewritten': 14, 'morphism_fires': {'norm_cascade': 12, 'weight_tie': 1}, 'end_to_end': {'max_abs_diff': 1.4781951904296875e-05, 'max_rel_diff': 1.0847653102289108e-06}, 'in_place': False, 'per_law': {'out_in_compose': {'matched': 10, 'grafted': 0, 'declined': 10, 'skipped': 0, 'reasons': {'no_improvement': 10}}, 'norm_cascade': {'matched': 12, 'grafted': 12, 'declined': 0, 'skipped': 0, 'reasons': {}}, 'weight_tie': {'matched': 79, 'grafted': 1, 'declined': 0, 'skipped': 78, 'reasons': {'node already rewritten': 78}}, 'kv_latent_share': {'matched': 12, 'grafted': 0, 'declined': 0, 'skipped': 12, 'reasons': {'node already rewritten': 12}}}, 'matches': {'out_in_compose:blocks.0+blocks.1': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.0.out_proj ∘ blocks.1.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.1+blocks.2': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.1.out_proj ∘ blocks.2.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.2+blocks.3': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.2.out_proj ∘ blocks.3.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.3+blocks.4': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.3.out_proj ∘ blocks.4.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.4+blocks.5': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.4.out_proj ∘ blocks.5.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.5+blocks.6': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.5.out_proj ∘ blocks.6.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.6+blocks.7': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.6.out_proj ∘ blocks.7.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.7+blocks.8': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.7.out_proj ∘ blocks.8.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.8+blocks.9': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.8.out_proj ∘ blocks.9.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'out_in_compose:blocks.9+blocks.10': {'status': 'declined', 'reason': 'no_improvement', 'boundary': 'chain', 'detail': 'blocks.9.out_proj ∘ blocks.10.in_projs', 'cost_before': 1840123904.0, 'cost_after': 1840123904.0}, 'norm_cascade:blocks.0': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.0 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 2.5213162493074435e-07}, 'norm_cascade:blocks.1': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.1 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 3.526764924367132e-07}, 'norm_cascade:blocks.2': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.2 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 1.5307266576774107e-07}, 'norm_cascade:blocks.3': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.3 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 1.0003477203431209e-07}, 'norm_cascade:blocks.4': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.4 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 7.714223203917101e-08}, 'norm_cascade:blocks.5': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.5 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 6.88405265583707e-08}, 'norm_cascade:blocks.6': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.6 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 1.1106595659707318e-07}, 'norm_cascade:blocks.7': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.7 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 1.004562925456669e-07}, 'norm_cascade:blocks.8': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.8 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 9.319301445253111e-08}, 'norm_cascade:blocks.9': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.9 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 1.4189854383033375e-07}, 'norm_cascade:blocks.10': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.10 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 1.4490870857287386e-07}, 'norm_cascade:blocks.11': {'status': 'grafted', 'boundary': 'intra', 'detail': 'blocks.11 pre-norm gain folds into in_projs', 'cost_before': 920064000.0, 'cost_after': 919965696.0, 'rel_diff': 1.3278309785702136e-07}, 'weight_tie:blocks.0': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:emb+head': {'status': 'grafted', 'boundary': 'tie', 'detail': 'emb~head: tied weights', 'rel_diff': 0.0, 'tied': [['emb__p_weight', 'head__p_weight']]}, 'weight_tie:blocks.0+blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.0+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.1+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.2+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.3+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.4+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5+blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5+blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.5+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.6+blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.6+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.6+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.6+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.6+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.7+blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.7+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.7+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.7+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.8+blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.8+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.8+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.9+blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.9+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'weight_tie:blocks.10+blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.0': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.1': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.10': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.11': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.2': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.3': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.4': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.5': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.6': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.7': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.8': {'status': 'skipped', 'reason': 'node already rewritten'}, 'kv_latent_share:blocks.9': {'status': 'skipped', 'reason': 'node already rewritten'}}, 'blocks': {}}, 'verify_fresh': {'passed': True, 'max_rel': 1.158577516358039e-06}, 'timing': {'eager': 235.963, 'morphism': 218.202, 'speedup': 1.0814}}} · `quick`=False

## Findings

| verdict | claim | headline | metric |
|---|---|---|---|
| **WIN** | stories15M: composer blocks lift to morphism signatures (incl. multi-input (h, cos, sin) attention blocks) | 8/8 blocks lifted | lift_fraction = 1 |
| **NEGATIVE** | stories15M: window_compose — 1 window match(es) on the real chain | matched 1, grafted 0 — guarded default stack: completed ({'window_compose:blocks.0+blocks.1+blocks.2+blocks.3+blocks.4': {'boundary': 'chain+chain+chain+chain', 'detail': 'blocks.0->…->blocks.4: 5-block out∘in window', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 663432192.0, 'cost_after': 663432192.0}}) | grafted = 0 |
| **INCONCLUSIVE** | stories15M: residual_reassoc — signature matches on the real graph | 0 candidates -> 0 grafted | grafted = 0 |
| **NEGATIVE** | stories15M: out_in_compose — signature matches on the real graph | 4 candidates -> 0 grafted (no_improvement×4) | grafted = 0 |
| **INCONCLUSIVE** | stories15M: residual_absorb — signature matches on the real graph | 0 candidates -> 0 grafted | grafted = 0 |
| **WIN** | stories15M: norm_cascade — signature matches on the real graph | 6 candidates -> 6 grafted | grafted = 6 |
| **WIN** | stories15M: weight_tie — signature matches on the real graph | 22 candidates -> 1 grafted (node already rewritten×21) | grafted = 1 |
| **INCONCLUSIVE** | stories15M: kv_latent_share — signature matches on the real graph | 6 candidates -> 0 grafted (node already rewritten×6) | grafted = 0 |
| **WIN** | stories15M: value-tied parameters reach weight_tie candidacy | 1 tied pair(s) in the module; 0 invisible to signature level | invisible_ties = 0 |
| **PARITY** | stories15M: every adjacent block boundary classifies | 3 opaque edge(s) — the lifted graph's reach stops at them | opaque_wires = 3 |
| **WIN** | stories15M: optimize_morphisms delivers a verified module and its wall-time delta vs eager | rewritten=8 rest_optimized=0 fresh_rel=1.4674530641600703e-06 1.230× | speedup_vs_eager = 1.2299 |
| **WIN** | stories110M: composer blocks lift to morphism signatures (incl. multi-input (h, cos, sin) attention blocks) | 14/14 blocks lifted | lift_fraction = 1 |
| **NEGATIVE** | stories110M: window_compose — 1 window match(es) on the real chain | matched 1, grafted 0 — guarded default stack: completed ({'window_compose:blocks.0+blocks.1+blocks.2+blocks.3': {'boundary': 'chain+chain+chain', 'detail': 'blocks.0->…->blocks.3: 4-block out∘in window', 'status': 'declined', 'reason': 'no_improvement', 'cost_before': 3680243712.0, 'cost_after': 3680243712.0}}) | grafted = 0 |
| **INCONCLUSIVE** | stories110M: residual_reassoc — signature matches on the real graph | 0 candidates -> 0 grafted | grafted = 0 |
| **NEGATIVE** | stories110M: out_in_compose — signature matches on the real graph | 10 candidates -> 0 grafted (no_improvement×10) | grafted = 0 |
| **INCONCLUSIVE** | stories110M: residual_absorb — signature matches on the real graph | 0 candidates -> 0 grafted | grafted = 0 |
| **WIN** | stories110M: norm_cascade — signature matches on the real graph | 12 candidates -> 12 grafted | grafted = 12 |
| **WIN** | stories110M: weight_tie — signature matches on the real graph | 79 candidates -> 1 grafted (node already rewritten×78) | grafted = 1 |
| **INCONCLUSIVE** | stories110M: kv_latent_share — signature matches on the real graph | 12 candidates -> 0 grafted (node already rewritten×12) | grafted = 0 |
| **WIN** | stories110M: value-tied parameters reach weight_tie candidacy | 1 tied pair(s) in the module; 0 invisible to signature level | invisible_ties = 0 |
| **PARITY** | stories110M: every adjacent block boundary classifies | 3 opaque edge(s) — the lifted graph's reach stops at them | opaque_wires = 3 |
| **WIN** | stories110M: optimize_morphisms delivers a verified module and its wall-time delta vs eager | rewritten=14 rest_optimized=0 fresh_rel=1.158577516358039e-06 1.081× | speedup_vs_eager = 1.0814 |

## Results

| case | model | seq | eager ms | morphism ms |
|---|---|---|---|---|
| stories15M@T64 | stories15M | 64 | 42.04 ± 0 | 34.18 ± 0 |
| stories110M@T64 | stories110M | 64 | 236 ± 18.39 | 218.2 ± 7.383 |

## Metrics

<details><summary><b>stories15M@T64</b> — raw metrics</summary>

```json
{
  "model": "stories15M",
  "ckpt": "/home/luis/.cache/catopt/stories15M.bin",
  "cfg": {
    "dim": 288,
    "hidden": 768,
    "n_layers": 6,
    "n_heads": 6,
    "vocab": 32000,
    "seq_len": 256
  },
  "seq": 64,
  "load_s": 11.0,
  "coverage": {
    "lift": {
      "n_nodes": 8,
      "n_lifted": 8,
      "n_opaque": 0,
      "input_kinds": {
        "activation": 8,
        "const_table": 12
      },
      "input_roles": {
        "stream": 8,
        "table": 12
      },
      "wire_kinds": {
        "opaque": 3,
        "chain": 4
      },
      "wires": [
        [
          "emb",
          "blocks.0",
          "opaque"
        ],
        [
          "blocks.0",
          "blocks.1",
          "chain"
        ],
        [
          "blocks.1",
          "blocks.2",
          "chain"
        ],
        [
          "blocks.2",
          "blocks.3",
          "chain"
        ],
        [
          "blocks.3",
          "blocks.4",
          "chain"
        ],
        [
          "blocks.4",
          "blocks.5",
          "opaque"
        ],
        [
          "blocks.5",
          "head",
          "opaque"
        ]
      ],
      "blocks": {
        "emb": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [],
            "out_proj": [],
            "norm": "none",
            "norm_affine": false,
            "norm_pre": false,
            "act": [],
            "residual": false,
            "shape": [
              [
                1,
                64
              ],
              [
                1,
                64,
                288
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "input",
                "kind": "activation",
                "role": "stream"
              }
            ]
          }
        },
        "blocks.0": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                288
              ],
              [
                64,
                288
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.1": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                288
              ],
              [
                64,
                288
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.2": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                288
              ],
              [
                64,
                288
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.3": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                288
              ],
              [
                64,
                288
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.4": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                288
              ],
              [
                64,
                288
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.5": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                288
              ],
              [
                64,
                288
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "head": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_weight"
            ],
            "out_proj": [
              "p_weight"
            ],
            "norm": "none",
            "norm_affine": false,
            "norm_pre": false,
            "act": [],
            "residual": false,
            "shape": [
              [
                64,
                288
              ],
              [
                64,
                32000
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "input",
                "kind": "activation",
                "role": "stream"
              }
            ]
          }
        }
      }
    },
    "candidacy": {
      "window_compose": {
        "matches": 1,
        "sites": [
          [
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4"
          ]
        ],
        "boundaries": [
          "chain+chain+chain+chain"
        ],
        "details": [
          "blocks.0->\u2026->blocks.4: 5-block out\u2218in window"
        ]
      },
      "residual_reassoc": {
        "matches": 0,
        "sites": [],
        "boundaries": [],
        "details": []
      },
      "out_in_compose": {
        "matches": 4,
        "sites": [
          [
            "blocks.0",
            "blocks.1"
          ],
          [
            "blocks.1",
            "blocks.2"
          ],
          [
            "blocks.2",
            "blocks.3"
          ],
          [
            "blocks.3",
            "blocks.4"
          ]
        ],
        "boundaries": [
          "chain",
          "chain",
          "chain",
          "chain"
        ],
        "details": [
          "blocks.0.out_proj \u2218 blocks.1.in_projs",
          "blocks.1.out_proj \u2218 blocks.2.in_projs",
          "blocks.2.out_proj \u2218 blocks.3.in_projs",
          "blocks.3.out_proj \u2218 blocks.4.in_projs"
        ]
      },
      "residual_absorb": {
        "matches": 0,
        "sites": [],
        "boundaries": [],
        "details": []
      },
      "norm_cascade": {
        "matches": 6,
        "sites": [
          [
            "blocks.0"
          ],
          [
            "blocks.1"
          ],
          [
            "blocks.2"
          ],
          [
            "blocks.3"
          ],
          [
            "blocks.4"
          ],
          [
            "blocks.5"
          ]
        ],
        "boundaries": [
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra"
        ],
        "details": [
          "blocks.0 pre-norm gain folds into in_projs",
          "blocks.1 pre-norm gain folds into in_projs",
          "blocks.2 pre-norm gain folds into in_projs",
          "blocks.3 pre-norm gain folds into in_projs",
          "blocks.4 pre-norm gain folds into in_projs",
          "blocks.5 pre-norm gain folds into in_projs"
        ]
      },
      "weight_tie": {
        "matches": 22,
        "sites": [
          [
            "blocks.0"
          ],
          [
            "blocks.1"
          ],
          [
            "blocks.2"
          ],
          [
            "blocks.3"
          ],
          [
            "blocks.4"
          ],
          [
            "blocks.5"
          ],
          [
            "emb",
            "head"
          ],
          [
            "blocks.0",
            "blocks.1"
          ],
          [
            "blocks.0",
            "blocks.2"
          ],
          [
            "blocks.0",
            "blocks.3"
          ],
          [
            "blocks.0",
            "blocks.4"
          ],
          [
            "blocks.0",
            "blocks.5"
          ],
          [
            "blocks.1",
            "blocks.2"
          ],
          [
            "blocks.1",
            "blocks.3"
          ],
          [
            "blocks.1",
            "blocks.4"
          ],
          [
            "blocks.1",
            "blocks.5"
          ],
          [
            "blocks.2",
            "blocks.3"
          ],
          [
            "blocks.2",
            "blocks.4"
          ],
          [
            "blocks.2",
            "blocks.5"
          ],
          [
            "blocks.3",
            "blocks.4"
          ],
          [
            "blocks.3",
            "blocks.5"
          ],
          [
            "blocks.4",
            "blocks.5"
          ]
        ],
        "boundaries": [
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie"
        ],
        "details": [
          "blocks.0: same-shape weights",
          "blocks.1: same-shape weights",
          "blocks.2: same-shape weights",
          "blocks.3: same-shape weights",
          "blocks.4: same-shape weights",
          "blocks.5: same-shape weights",
          "emb~head: tied weights",
          "blocks.0~blocks.1: tied weights",
          "blocks.0~blocks.2: tied weights",
          "blocks.0~blocks.3: tied weights",
          "blocks.0~blocks.4: tied weights",
          "blocks.0~blocks.5: tied weights",
          "blocks.1~blocks.2: tied weights",
          "blocks.1~blocks.3: tied weights",
          "blocks.1~blocks.4: tied weights",
          "blocks.1~blocks.5: tied weights",
          "blocks.2~blocks.3: tied weights",
          "blocks.2~blocks.4: tied weights",
          "blocks.2~blocks.5: tied weights",
          "blocks.3~blocks.4: tied weights",
          "blocks.3~blocks.5: tied weights",
          "blocks.4~blocks.5: tied weights"
        ]
      },
      "kv_latent_share": {
        "matches": 6,
        "sites": [
          [
            "blocks.0"
          ],
          [
            "blocks.1"
          ],
          [
            "blocks.2"
          ],
          [
            "blocks.3"
          ],
          [
            "blocks.4"
          ],
          [
            "blocks.5"
          ]
        ],
        "boundaries": [
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra"
        ],
        "details": [
          "blocks.0: shared-data kv projections",
          "blocks.1: shared-data kv projections",
          "blocks.2: shared-data kv projections",
          "blocks.3: shared-data kv projections",
          "blocks.4: shared-data kv projections",
          "blocks.5: shared-data kv projections"
        ]
      }
    },
    "seconds": 7.63
  },
  "gaps": {
    "tied_param_pairs": [
      {
        "a": "emb.weight",
        "b": "head.weight",
        "shape": [
          32000,
          288
        ]
      }
    ],
    "norm_ops_vs_sig": {
      "emb": {
        "norm_ops": {},
        "sig_norm": "none"
      },
      "blocks.0": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.1": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.2": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.3": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.4": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.5": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "head": {
        "norm_ops": {},
        "sig_norm": "none"
      }
    },
    "opaque_wires": [
      [
        "emb",
        "blocks.0",
        "opaque"
      ],
      [
        "blocks.4",
        "blocks.5",
        "opaque"
      ],
      [
        "blocks.5",
        "head",
        "opaque"
      ]
    ]
  },
  "default_stack": {
    "laws": "all",
    "optimize_rest": true,
    "exit": 0,
    "seconds": 224.8,
    "died_at": [],
    "progress_tail": [
      [
        "DONE",
        "",
        ""
      ]
    ],
    "stderr_tail": "",
    "stats": {
      "morphism": true,
      "n_blocks": 8,
      "n_lifted": 8,
      "sigs": {
        "emb": {
          "in_projs": [],
          "out_proj": [],
          "tables": [
            "p_weight"
          ],
          "norm": "none",
          "norm_affine": false,
          "norm_pre": false,
          "act": [],
          "residual": false,
          "shape": [
            [
              1,
              64
            ],
            [
              1,
              64,
              288
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "input",
              "kind": "activation",
              "role": "stream"
            }
          ]
        },
        "blocks.0": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              288
            ],
            [
              64,
              288
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.1": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              288
            ],
            [
              64,
              288
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.2": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              288
            ],
            [
              64,
              288
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.3": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              288
            ],
            [
              64,
              288
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.4": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              288
            ],
            [
              64,
              288
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.5": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              288
            ],
            [
              64,
              288
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "head": {
          "in_projs": [
            "p_weight"
          ],
          "out_proj": [
            "p_weight"
          ],
          "tables": [],
          "norm": "none",
          "norm_affine": false,
          "norm_pre": false,
          "act": [],
          "residual": false,
          "shape": [
            [
              64,
              288
            ],
            [
              64,
              32000
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "input",
              "kind": "activation",
              "role": "stream"
            }
          ]
        }
      },
      "wires": [
        [
          "emb",
          "blocks.0",
          "opaque"
        ],
        [
          "blocks.0",
          "blocks.1",
          "chain"
        ],
        [
          "blocks.1",
          "blocks.2",
          "chain"
        ],
        [
          "blocks.2",
          "blocks.3",
          "chain"
        ],
        [
          "blocks.3",
          "blocks.4",
          "chain"
        ],
        [
          "blocks.4",
          "blocks.5",
          "opaque"
        ],
        [
          "blocks.5",
          "head",
          "opaque"
        ]
      ],
      "matches": {
        "window_compose:blocks.0+blocks.1+blocks.2+blocks.3+blocks.4": {
          "boundary": "chain+chain+chain+chain",
          "detail": "blocks.0->\u2026->blocks.4: 5-block out\u2218in window",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 663432192.0,
          "cost_after": 663432192.0
        },
        "out_in_compose:blocks.0+blocks.1": {
          "boundary": "chain",
          "detail": "blocks.0.out_proj \u2218 blocks.1.in_projs",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 265374720.0,
          "cost_after": 265374720.0
        },
        "out_in_compose:blocks.1+blocks.2": {
          "boundary": "chain",
          "detail": "blocks.1.out_proj \u2218 blocks.2.in_projs",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 265374720.0,
          "cost_after": 265374720.0
        },
        "out_in_compose:blocks.2+blocks.3": {
          "boundary": "chain",
          "detail": "blocks.2.out_proj \u2218 blocks.3.in_projs",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 265374720.0,
          "cost_after": 265374720.0
        },
        "out_in_compose:blocks.3+blocks.4": {
          "boundary": "chain",
          "detail": "blocks.3.out_proj \u2218 blocks.4.in_projs",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 265374720.0,
          "cost_after": 265374720.0
        },
        "norm_cascade:blocks.0": {
          "boundary": "intra",
          "detail": "blocks.0 pre-norm gain folds into in_projs",
          "status": "grafted",
          "cost_before": 132688896.0,
          "cost_after": 132652032.0,
          "rel_diff": 2.6346184484560376e-07
        },
        "norm_cascade:blocks.1": {
          "boundary": "intra",
          "detail": "blocks.1 pre-norm gain folds into in_projs",
          "status": "grafted",
          "cost_before": 132688896.0,
          "cost_after": 132652032.0,
          "rel_diff": 2.647190394666863e-07
        },
        "norm_cascade:blocks.2": {
          "boundary": "intra",
          "detail": "blocks.2 pre-norm gain folds into in_projs",
          "status": "grafted",
          "cost_before": 132688896.0,
          "cost_after": 132652032.0,
          "rel_diff": 2.0895082609941918e-07
        },
        "norm_cascade:blocks.3": {
          "boundary": "intra",
          "detail": "blocks.3 pre-norm gain folds into in_projs",
          "status": "grafted",
          "cost_before": 132688896.0,
          "cost_after": 132652032.0,
          "rel_diff": 1.0293398914506104e-07
        },
        "norm_cascade:blocks.4": {
          "boundary": "intra",
          "detail": "blocks.4 pre-norm gain folds into in_projs",
          "status": "grafted",
          "cost_before": 132688896.0,
          "cost_after": 132652032.0,
          "rel_diff": 1.0775967726300525e-07
        },
        "norm_cascade:blocks.5": {
          "boundary": "intra",
          "detail": "blocks.5 pre-norm gain folds into in_projs",
          "status": "grafted",
          "cost_before": 132688896.0,
          "cost_after": 132652032.0,
          "rel_diff": 2.050507441414449e-07
        },
        "weight_tie:blocks.0": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.1": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.2": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.3": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.4": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.5": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:emb+head": {
          "boundary": "tie",
          "detail": "emb~head: tied weights",
          "status": "grafted",
          "tied": [
            [
              "emb__p_weight",
              "head__p_weight"
            ]
          ],
          "rel_diff": 0.0
        },
        "weight_tie:blocks.0+blocks.1": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.0+blocks.2": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.0+blocks.3": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.0+blocks.4": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.0+blocks.5": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.1+blocks.2": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.1+blocks.3": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.1+blocks.4": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.1+blocks.5": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.2+blocks.3": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.2+blocks.4": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.2+blocks.5": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.3+blocks.4": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.3+blocks.5": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "weight_tie:blocks.4+blocks.5": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "kv_latent_share:blocks.0": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "kv_latent_share:blocks.1": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "kv_latent_share:blocks.2": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "kv_latent_share:blocks.3": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "kv_latent_share:blocks.4": {
          "status": "skipped",
          "reason": "node already rewritten"
        },
        "kv_latent_share:blocks.5": {
          "status": "skipped",
          "reason": "node already rewritten"
        }
      },
      "morphism_fires": {
        "norm_cascade": 6,
        "weight_tie": 1
      },
      "n_rewritten": 8,
      "blocks": {
        "emb": {
          "status": "rewritten",
          "law": "weight_tie"
        },
        "blocks.0": {
          "status": "rewritten",
          "law": "norm_cascade"
        },
        "blocks.1": {
          "status": "rewritten",
          "law": "norm_cascade"
        },
        "blocks.2": {
          "status": "rewritten",
          "law": "norm_cascade"
        },
        "blocks.3": {
          "status": "rewritten",
          "law": "norm_cascade"
        },
        "blocks.4": {
          "status": "rewritten",
          "law": "norm_cascade"
        },
        "blocks.5": {
          "status": "rewritten",
          "law": "norm_cascade"
        },
        "head": {
          "status": "rewritten",
          "law": "weight_tie"
        }
      },
      "in_place": false,
      "shared_params": true,
      "end_to_end": {
        "max_abs_diff": 2.47955322265625e-05,
        "max_rel_diff": 1.7041540118787826e-06
      },
      "wall_time_s": 186.33752822875977
    },
    "outcome": "completed"
  },
  "delivery": {
    "laws": [
      "residual_reassoc",
      "out_in_compose",
      "residual_absorb",
      "norm_cascade",
      "weight_tie",
      "kv_latent_share"
    ],
    "optimize_rest": true,
    "window_excluded": true,
    "window_exclusion_reason": "window match declined under the guarded run",
    "seconds": 3.3,
    "n_rewritten": 8,
    "morphism_fires": {
      "norm_cascade": 6,
      "weight_tie": 1
    },
    "end_to_end": {
      "max_abs_diff": 2.47955322265625e-05,
      "max_rel_diff": 1.7041540118787826e-06
    },
    "in_place": false,
    "per_law": {
      "out_in_compose": {
        "matched": 4,
        "grafted": 0,
        "declined": 4,
        "skipped": 0,
        "reasons": {
          "no_improvement": 4
        }
      },
      "norm_cascade": {
        "matched": 6,
        "grafted": 6,
        "declined": 0,
        "skipped": 0,
        "reasons": {}
      },
      "weight_tie": {
        "matched": 22,
        "grafted": 1,
        "declined": 0,
        "skipped": 21,
        "reasons": {
          "node already rewritten": 21
        }
      },
      "kv_latent_share": {
        "matched": 6,
        "grafted": 0,
        "declined": 0,
        "skipped": 6,
        "reasons": {
          "node already rewritten": 6
        }
      }
    },
    "matches": {
      "out_in_compose:blocks.0+blocks.1": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.0.out_proj \u2218 blocks.1.in_projs",
        "cost_before": 265374720.0,
        "cost_after": 265374720.0
      },
      "out_in_compose:blocks.1+blocks.2": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.1.out_proj \u2218 blocks.2.in_projs",
        "cost_before": 265374720.0,
        "cost_after": 265374720.0
      },
      "out_in_compose:blocks.2+blocks.3": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.2.out_proj \u2218 blocks.3.in_projs",
        "cost_before": 265374720.0,
        "cost_after": 265374720.0
      },
      "out_in_compose:blocks.3+blocks.4": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.3.out_proj \u2218 blocks.4.in_projs",
        "cost_before": 265374720.0,
        "cost_after": 265374720.0
      },
      "norm_cascade:blocks.0": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.0 pre-norm gain folds into in_projs",
        "cost_before": 132688896.0,
        "cost_after": 132652032.0,
        "rel_diff": 2.6346184484560376e-07
      },
      "norm_cascade:blocks.1": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.1 pre-norm gain folds into in_projs",
        "cost_before": 132688896.0,
        "cost_after": 132652032.0,
        "rel_diff": 2.647190394666863e-07
      },
      "norm_cascade:blocks.2": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.2 pre-norm gain folds into in_projs",
        "cost_before": 132688896.0,
        "cost_after": 132652032.0,
        "rel_diff": 2.0895082609941918e-07
      },
      "norm_cascade:blocks.3": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.3 pre-norm gain folds into in_projs",
        "cost_before": 132688896.0,
        "cost_after": 132652032.0,
        "rel_diff": 1.0293398914506104e-07
      },
      "norm_cascade:blocks.4": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.4 pre-norm gain folds into in_projs",
        "cost_before": 132688896.0,
        "cost_after": 132652032.0,
        "rel_diff": 1.0775967726300525e-07
      },
      "norm_cascade:blocks.5": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.5 pre-norm gain folds into in_projs",
        "cost_before": 132688896.0,
        "cost_after": 132652032.0,
        "rel_diff": 2.050507441414449e-07
      },
      "weight_tie:blocks.0": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:emb+head": {
        "status": "grafted",
        "boundary": "tie",
        "detail": "emb~head: tied weights",
        "rel_diff": 0.0,
        "tied": [
          [
            "emb__p_weight",
            "head__p_weight"
          ]
        ]
      },
      "weight_tie:blocks.0+blocks.1": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.0": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.1": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      }
    },
    "blocks": {
      "emb": {
        "status": "rewritten",
        "law": "weight_tie"
      },
      "blocks.0": {
        "status": "rewritten",
        "law": "norm_cascade"
      },
      "blocks.1": {
        "status": "rewritten",
        "law": "norm_cascade"
      },
      "blocks.2": {
        "status": "rewritten",
        "law": "norm_cascade"
      },
      "blocks.3": {
        "status": "rewritten",
        "law": "norm_cascade"
      },
      "blocks.4": {
        "status": "rewritten",
        "law": "norm_cascade"
      },
      "blocks.5": {
        "status": "rewritten",
        "law": "norm_cascade"
      },
      "head": {
        "status": "rewritten",
        "law": "weight_tie"
      }
    }
  },
  "verify_fresh": {
    "passed": true,
    "max_rel": 1.4674530641600703e-06
  },
  "timing": {
    "eager": 42.035,
    "morphism": 34.177,
    "speedup": 1.2299
  }
}
```

</details>

<details><summary><b>stories110M@T64</b> — raw metrics</summary>

```json
{
  "model": "stories110M",
  "ckpt": "/home/luis/.cache/catopt/stories110M.bin",
  "cfg": {
    "dim": 768,
    "hidden": 2048,
    "n_layers": 12,
    "n_heads": 12,
    "vocab": 32000,
    "seq_len": 1024
  },
  "seq": 64,
  "load_s": 1.5,
  "coverage": {
    "lift": {
      "n_nodes": 14,
      "n_lifted": 14,
      "n_opaque": 0,
      "input_kinds": {
        "activation": 14,
        "const_table": 24
      },
      "input_roles": {
        "stream": 14,
        "table": 24
      },
      "wire_kinds": {
        "opaque": 3,
        "chain": 10
      },
      "wires": [
        [
          "emb",
          "blocks.0",
          "opaque"
        ],
        [
          "blocks.0",
          "blocks.1",
          "chain"
        ],
        [
          "blocks.1",
          "blocks.2",
          "chain"
        ],
        [
          "blocks.2",
          "blocks.3",
          "chain"
        ],
        [
          "blocks.3",
          "blocks.4",
          "chain"
        ],
        [
          "blocks.4",
          "blocks.5",
          "chain"
        ],
        [
          "blocks.5",
          "blocks.6",
          "chain"
        ],
        [
          "blocks.6",
          "blocks.7",
          "chain"
        ],
        [
          "blocks.7",
          "blocks.8",
          "chain"
        ],
        [
          "blocks.8",
          "blocks.9",
          "chain"
        ],
        [
          "blocks.9",
          "blocks.10",
          "chain"
        ],
        [
          "blocks.10",
          "blocks.11",
          "opaque"
        ],
        [
          "blocks.11",
          "head",
          "opaque"
        ]
      ],
      "blocks": {
        "emb": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [],
            "out_proj": [],
            "norm": "none",
            "norm_affine": false,
            "norm_pre": false,
            "act": [],
            "residual": false,
            "shape": [
              [
                1,
                64
              ],
              [
                1,
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "input",
                "kind": "activation",
                "role": "stream"
              }
            ]
          }
        },
        "blocks.0": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.1": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.2": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.3": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.4": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.5": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.6": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.7": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.8": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.9": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.10": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "blocks.11": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_wv_weight",
              "p_wk_weight",
              "p_wq_weight"
            ],
            "out_proj": [
              "p_w2_weight"
            ],
            "norm": "rms",
            "norm_affine": true,
            "norm_pre": true,
            "act": [
              "sdpa",
              "silu"
            ],
            "residual": true,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                768
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "h",
                "kind": "activation",
                "role": "stream"
              },
              {
                "index": 1,
                "name": "cos",
                "kind": "const_table",
                "role": "table"
              },
              {
                "index": 2,
                "name": "sin",
                "kind": "const_table",
                "role": "table"
              }
            ]
          }
        },
        "head": {
          "opaque": false,
          "note": null,
          "calls": 1,
          "sig": {
            "in_projs": [
              "p_weight"
            ],
            "out_proj": [
              "p_weight"
            ],
            "norm": "none",
            "norm_affine": false,
            "norm_pre": false,
            "act": [],
            "residual": false,
            "shape": [
              [
                64,
                768
              ],
              [
                64,
                32000
              ]
            ],
            "inputs": [
              {
                "index": 0,
                "name": "input",
                "kind": "activation",
                "role": "stream"
              }
            ]
          }
        }
      }
    },
    "candidacy": {
      "window_compose": {
        "matches": 1,
        "sites": [
          [
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",
            "blocks.6",
            "blocks.7",
            "blocks.8",
            "blocks.9",
            "blocks.10"
          ]
        ],
        "boundaries": [
          "chain+chain+chain+chain+chain+chain+chain+chain+chain+chain"
        ],
        "details": [
          "blocks.0->\u2026->blocks.10: 11-block out\u2218in window"
        ]
      },
      "residual_reassoc": {
        "matches": 0,
        "sites": [],
        "boundaries": [],
        "details": []
      },
      "out_in_compose": {
        "matches": 10,
        "sites": [
          [
            "blocks.0",
            "blocks.1"
          ],
          [
            "blocks.1",
            "blocks.2"
          ],
          [
            "blocks.2",
            "blocks.3"
          ],
          [
            "blocks.3",
            "blocks.4"
          ],
          [
            "blocks.4",
            "blocks.5"
          ],
          [
            "blocks.5",
            "blocks.6"
          ],
          [
            "blocks.6",
            "blocks.7"
          ],
          [
            "blocks.7",
            "blocks.8"
          ],
          [
            "blocks.8",
            "blocks.9"
          ],
          [
            "blocks.9",
            "blocks.10"
          ]
        ],
        "boundaries": [
          "chain",
          "chain",
          "chain",
          "chain",
          "chain",
          "chain",
          "chain",
          "chain",
          "chain",
          "chain"
        ],
        "details": [
          "blocks.0.out_proj \u2218 blocks.1.in_projs",
          "blocks.1.out_proj \u2218 blocks.2.in_projs",
          "blocks.2.out_proj \u2218 blocks.3.in_projs",
          "blocks.3.out_proj \u2218 blocks.4.in_projs",
          "blocks.4.out_proj \u2218 blocks.5.in_projs",
          "blocks.5.out_proj \u2218 blocks.6.in_projs",
          "blocks.6.out_proj \u2218 blocks.7.in_projs",
          "blocks.7.out_proj \u2218 blocks.8.in_projs",
          "blocks.8.out_proj \u2218 blocks.9.in_projs",
          "blocks.9.out_proj \u2218 blocks.10.in_projs"
        ]
      },
      "residual_absorb": {
        "matches": 0,
        "sites": [],
        "boundaries": [],
        "details": []
      },
      "norm_cascade": {
        "matches": 12,
        "sites": [
          [
            "blocks.0"
          ],
          [
            "blocks.1"
          ],
          [
            "blocks.2"
          ],
          [
            "blocks.3"
          ],
          [
            "blocks.4"
          ],
          [
            "blocks.5"
          ],
          [
            "blocks.6"
          ],
          [
            "blocks.7"
          ],
          [
            "blocks.8"
          ],
          [
            "blocks.9"
          ],
          [
            "blocks.10"
          ],
          [
            "blocks.11"
          ]
        ],
        "boundaries": [
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra"
        ],
        "details": [
          "blocks.0 pre-norm gain folds into in_projs",
          "blocks.1 pre-norm gain folds into in_projs",
          "blocks.2 pre-norm gain folds into in_projs",
          "blocks.3 pre-norm gain folds into in_projs",
          "blocks.4 pre-norm gain folds into in_projs",
          "blocks.5 pre-norm gain folds into in_projs",
          "blocks.6 pre-norm gain folds into in_projs",
          "blocks.7 pre-norm gain folds into in_projs",
          "blocks.8 pre-norm gain folds into in_projs",
          "blocks.9 pre-norm gain folds into in_projs",
          "blocks.10 pre-norm gain folds into in_projs",
          "blocks.11 pre-norm gain folds into in_projs"
        ]
      },
      "weight_tie": {
        "matches": 79,
        "sites": [
          [
            "blocks.0"
          ],
          [
            "blocks.1"
          ],
          [
            "blocks.2"
          ],
          [
            "blocks.3"
          ],
          [
            "blocks.4"
          ],
          [
            "blocks.5"
          ],
          [
            "blocks.6"
          ],
          [
            "blocks.7"
          ],
          [
            "blocks.8"
          ],
          [
            "blocks.9"
          ],
          [
            "blocks.10"
          ],
          [
            "blocks.11"
          ],
          [
            "emb",
            "head"
          ],
          [
            "blocks.0",
            "blocks.1"
          ],
          [
            "blocks.0",
            "blocks.2"
          ],
          [
            "blocks.0",
            "blocks.3"
          ],
          [
            "blocks.0",
            "blocks.4"
          ],
          [
            "blocks.0",
            "blocks.5"
          ],
          [
            "blocks.0",
            "blocks.6"
          ],
          [
            "blocks.0",
            "blocks.7"
          ],
          [
            "blocks.0",
            "blocks.8"
          ],
          [
            "blocks.0",
            "blocks.9"
          ],
          [
            "blocks.0",
            "blocks.10"
          ],
          [
            "blocks.0",
            "blocks.11"
          ],
          [
            "blocks.1",
            "blocks.2"
          ],
          [
            "blocks.1",
            "blocks.3"
          ],
          [
            "blocks.1",
            "blocks.4"
          ],
          [
            "blocks.1",
            "blocks.5"
          ],
          [
            "blocks.1",
            "blocks.6"
          ],
          [
            "blocks.1",
            "blocks.7"
          ],
          [
            "blocks.1",
            "blocks.8"
          ],
          [
            "blocks.1",
            "blocks.9"
          ],
          [
            "blocks.1",
            "blocks.10"
          ],
          [
            "blocks.1",
            "blocks.11"
          ],
          [
            "blocks.2",
            "blocks.3"
          ],
          [
            "blocks.2",
            "blocks.4"
          ],
          [
            "blocks.2",
            "blocks.5"
          ],
          [
            "blocks.2",
            "blocks.6"
          ],
          [
            "blocks.2",
            "blocks.7"
          ],
          [
            "blocks.2",
            "blocks.8"
          ],
          [
            "blocks.2",
            "blocks.9"
          ],
          [
            "blocks.2",
            "blocks.10"
          ],
          [
            "blocks.2",
            "blocks.11"
          ],
          [
            "blocks.3",
            "blocks.4"
          ],
          [
            "blocks.3",
            "blocks.5"
          ],
          [
            "blocks.3",
            "blocks.6"
          ],
          [
            "blocks.3",
            "blocks.7"
          ],
          [
            "blocks.3",
            "blocks.8"
          ],
          [
            "blocks.3",
            "blocks.9"
          ],
          [
            "blocks.3",
            "blocks.10"
          ],
          [
            "blocks.3",
            "blocks.11"
          ],
          [
            "blocks.4",
            "blocks.5"
          ],
          [
            "blocks.4",
            "blocks.6"
          ],
          [
            "blocks.4",
            "blocks.7"
          ],
          [
            "blocks.4",
            "blocks.8"
          ],
          [
            "blocks.4",
            "blocks.9"
          ],
          [
            "blocks.4",
            "blocks.10"
          ],
          [
            "blocks.4",
            "blocks.11"
          ],
          [
            "blocks.5",
            "blocks.6"
          ],
          [
            "blocks.5",
            "blocks.7"
          ],
          [
            "blocks.5",
            "blocks.8"
          ],
          [
            "blocks.5",
            "blocks.9"
          ],
          [
            "blocks.5",
            "blocks.10"
          ],
          [
            "blocks.5",
            "blocks.11"
          ],
          [
            "blocks.6",
            "blocks.7"
          ],
          [
            "blocks.6",
            "blocks.8"
          ],
          [
            "blocks.6",
            "blocks.9"
          ],
          [
            "blocks.6",
            "blocks.10"
          ],
          [
            "blocks.6",
            "blocks.11"
          ],
          [
            "blocks.7",
            "blocks.8"
          ],
          [
            "blocks.7",
            "blocks.9"
          ],
          [
            "blocks.7",
            "blocks.10"
          ],
          [
            "blocks.7",
            "blocks.11"
          ],
          [
            "blocks.8",
            "blocks.9"
          ],
          [
            "blocks.8",
            "blocks.10"
          ],
          [
            "blocks.8",
            "blocks.11"
          ],
          [
            "blocks.9",
            "blocks.10"
          ],
          [
            "blocks.9",
            "blocks.11"
          ],
          [
            "blocks.10",
            "blocks.11"
          ]
        ],
        "boundaries": [
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie",
          "tie"
        ],
        "details": [
          "blocks.0: same-shape weights",
          "blocks.1: same-shape weights",
          "blocks.2: same-shape weights",
          "blocks.3: same-shape weights",
          "blocks.4: same-shape weights",
          "blocks.5: same-shape weights",
          "blocks.6: same-shape weights",
          "blocks.7: same-shape weights",
          "blocks.8: same-shape weights",
          "blocks.9: same-shape weights",
          "blocks.10: same-shape weights",
          "blocks.11: same-shape weights",
          "emb~head: tied weights",
          "blocks.0~blocks.1: tied weights",
          "blocks.0~blocks.2: tied weights",
          "blocks.0~blocks.3: tied weights",
          "blocks.0~blocks.4: tied weights",
          "blocks.0~blocks.5: tied weights",
          "blocks.0~blocks.6: tied weights",
          "blocks.0~blocks.7: tied weights",
          "blocks.0~blocks.8: tied weights",
          "blocks.0~blocks.9: tied weights",
          "blocks.0~blocks.10: tied weights",
          "blocks.0~blocks.11: tied weights",
          "blocks.1~blocks.2: tied weights",
          "blocks.1~blocks.3: tied weights",
          "blocks.1~blocks.4: tied weights",
          "blocks.1~blocks.5: tied weights",
          "blocks.1~blocks.6: tied weights",
          "blocks.1~blocks.7: tied weights",
          "blocks.1~blocks.8: tied weights",
          "blocks.1~blocks.9: tied weights",
          "blocks.1~blocks.10: tied weights",
          "blocks.1~blocks.11: tied weights",
          "blocks.2~blocks.3: tied weights",
          "blocks.2~blocks.4: tied weights",
          "blocks.2~blocks.5: tied weights",
          "blocks.2~blocks.6: tied weights",
          "blocks.2~blocks.7: tied weights",
          "blocks.2~blocks.8: tied weights",
          "blocks.2~blocks.9: tied weights",
          "blocks.2~blocks.10: tied weights",
          "blocks.2~blocks.11: tied weights",
          "blocks.3~blocks.4: tied weights",
          "blocks.3~blocks.5: tied weights",
          "blocks.3~blocks.6: tied weights",
          "blocks.3~blocks.7: tied weights",
          "blocks.3~blocks.8: tied weights",
          "blocks.3~blocks.9: tied weights",
          "blocks.3~blocks.10: tied weights",
          "blocks.3~blocks.11: tied weights",
          "blocks.4~blocks.5: tied weights",
          "blocks.4~blocks.6: tied weights",
          "blocks.4~blocks.7: tied weights",
          "blocks.4~blocks.8: tied weights",
          "blocks.4~blocks.9: tied weights",
          "blocks.4~blocks.10: tied weights",
          "blocks.4~blocks.11: tied weights",
          "blocks.5~blocks.6: tied weights",
          "blocks.5~blocks.7: tied weights",
          "blocks.5~blocks.8: tied weights",
          "blocks.5~blocks.9: tied weights",
          "blocks.5~blocks.10: tied weights",
          "blocks.5~blocks.11: tied weights",
          "blocks.6~blocks.7: tied weights",
          "blocks.6~blocks.8: tied weights",
          "blocks.6~blocks.9: tied weights",
          "blocks.6~blocks.10: tied weights",
          "blocks.6~blocks.11: tied weights",
          "blocks.7~blocks.8: tied weights",
          "blocks.7~blocks.9: tied weights",
          "blocks.7~blocks.10: tied weights",
          "blocks.7~blocks.11: tied weights",
          "blocks.8~blocks.9: tied weights",
          "blocks.8~blocks.10: tied weights",
          "blocks.8~blocks.11: tied weights",
          "blocks.9~blocks.10: tied weights",
          "blocks.9~blocks.11: tied weights",
          "blocks.10~blocks.11: tied weights"
        ]
      },
      "kv_latent_share": {
        "matches": 12,
        "sites": [
          [
            "blocks.0"
          ],
          [
            "blocks.1"
          ],
          [
            "blocks.10"
          ],
          [
            "blocks.11"
          ],
          [
            "blocks.2"
          ],
          [
            "blocks.3"
          ],
          [
            "blocks.4"
          ],
          [
            "blocks.5"
          ],
          [
            "blocks.6"
          ],
          [
            "blocks.7"
          ],
          [
            "blocks.8"
          ],
          [
            "blocks.9"
          ]
        ],
        "boundaries": [
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra",
          "intra"
        ],
        "details": [
          "blocks.0: shared-data kv projections",
          "blocks.1: shared-data kv projections",
          "blocks.10: shared-data kv projections",
          "blocks.11: shared-data kv projections",
          "blocks.2: shared-data kv projections",
          "blocks.3: shared-data kv projections",
          "blocks.4: shared-data kv projections",
          "blocks.5: shared-data kv projections",
          "blocks.6: shared-data kv projections",
          "blocks.7: shared-data kv projections",
          "blocks.8: shared-data kv projections",
          "blocks.9: shared-data kv projections"
        ]
      }
    },
    "seconds": 1.83
  },
  "gaps": {
    "tied_param_pairs": [
      {
        "a": "emb.weight",
        "b": "head.weight",
        "shape": [
          32000,
          768
        ]
      }
    ],
    "norm_ops_vs_sig": {
      "emb": {
        "norm_ops": {},
        "sig_norm": "none"
      },
      "blocks.0": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.1": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.2": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.3": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.4": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.5": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.6": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.7": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.8": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.9": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.10": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "blocks.11": {
        "norm_ops": {
          "rms_norm": 2
        },
        "sig_norm": "rms"
      },
      "head": {
        "norm_ops": {},
        "sig_norm": "none"
      }
    },
    "opaque_wires": [
      [
        "emb",
        "blocks.0",
        "opaque"
      ],
      [
        "blocks.10",
        "blocks.11",
        "opaque"
      ],
      [
        "blocks.11",
        "head",
        "opaque"
      ]
    ]
  },
  "default_stack": {
    "laws": "all",
    "optimize_rest": false,
    "exit": 0,
    "seconds": 46.7,
    "died_at": [],
    "progress_tail": [
      [
        "DONE",
        "",
        ""
      ]
    ],
    "stderr_tail": "",
    "stats": {
      "morphism": true,
      "n_blocks": 14,
      "n_lifted": 5,
      "sigs": {
        "emb": {
          "in_projs": [],
          "out_proj": [],
          "tables": [
            "p_weight"
          ],
          "norm": "none",
          "norm_affine": false,
          "norm_pre": false,
          "act": [],
          "residual": false,
          "shape": [
            [
              1,
              64
            ],
            [
              1,
              64,
              768
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "input",
              "kind": "activation",
              "role": "stream"
            }
          ]
        },
        "blocks.0": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              768
            ],
            [
              64,
              768
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.1": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              768
            ],
            [
              64,
              768
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.2": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              768
            ],
            [
              64,
              768
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.3": {
          "in_projs": [
            "p_wv_weight",
            "p_wk_weight",
            "p_wq_weight"
          ],
          "out_proj": [
            "p_w2_weight"
          ],
          "tables": [],
          "norm": "rms",
          "norm_affine": true,
          "norm_pre": true,
          "act": [
            "sdpa",
            "silu"
          ],
          "residual": true,
          "shape": [
            [
              64,
              768
            ],
            [
              64,
              768
            ]
          ],
          "inputs": [
            {
              "index": 0,
              "name": "h",
              "kind": "activation",
              "role": "stream"
            },
            {
              "index": 1,
              "name": "cos",
              "kind": "const_table",
              "role": "table"
            },
            {
              "index": 2,
              "name": "sin",
              "kind": "const_table",
              "role": "table"
            }
          ]
        },
        "blocks.4": null,
        "blocks.5": null,
        "blocks.6": null,
        "blocks.7": null,
        "blocks.8": null,
        "blocks.9": null,
        "blocks.10": null,
        "blocks.11": null,
        "head": null
      },
      "wires": [
        [
          "emb",
          "blocks.0",
          "opaque"
        ],
        [
          "blocks.0",
          "blocks.1",
          "chain"
        ],
        [
          "blocks.1",
          "blocks.2",
          "chain"
        ],
        [
          "blocks.2",
          "blocks.3",
          "chain"
        ],
        [
          "blocks.3",
          "blocks.4",
          "chain"
        ],
        [
          "blocks.4",
          "blocks.5",
          "opaque"
        ],
        [
          "blocks.5",
          "blocks.6",
          "opaque"
        ],
        [
          "blocks.6",
          "blocks.7",
          "opaque"
        ],
        [
          "blocks.7",
          "blocks.8",
          "opaque"
        ],
        [
          "blocks.8",
          "blocks.9",
          "opaque"
        ],
        [
          "blocks.9",
          "blocks.10",
          "opaque"
        ],
        [
          "blocks.10",
          "blocks.11",
          "opaque"
        ],
        [
          "blocks.11",
          "head",
          "opaque"
        ]
      ],
      "matches": {
        "window_compose:blocks.0+blocks.1+blocks.2+blocks.3": {
          "boundary": "chain+chain+chain",
          "detail": "blocks.0->\u2026->blocks.3: 4-block out\u2218in window",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 3680243712.0,
          "cost_after": 3680243712.0
        },
        "out_in_compose:blocks.0+blocks.1": {
          "boundary": "chain",
          "detail": "blocks.0.out_proj \u2218 blocks.1.in_projs",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 1840123904.0,
          "cost_after": 1840123904.0
        },
        "out_in_compose:blocks.1+blocks.2": {
          "boundary": "chain",
          "detail": "blocks.1.out_proj \u2218 blocks.2.in_projs",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 1840123904.0,
          "cost_after": 1840123904.0
        },
        "out_in_compose:blocks.2+blocks.3": {
          "boundary": "chain",
          "detail": "blocks.2.out_proj \u2218 blocks.3.in_projs",
          "status": "declined",
          "reason": "no_improvement",
          "cost_before": 1840123904.0,
          "cost_after": 1840123904.0
        },
        "norm_cascade:blocks.0": {
          "status": "declined",
          "reason": "error",
          "error": "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"
        },
        "norm_cascade:blocks.1": {
          "status": "declined",
          "reason": "error",
          "error": "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"
        },
        "norm_cascade:blocks.2": {
          "status": "declined",
          "reason": "error",
          "error": "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"
        },
        "norm_cascade:blocks.3": {
          "status": "declined",
          "reason": "error",
          "error": "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 6291456 bytes. Error code 12 (Cannot allocate memory)"
        },
        "weight_tie:blocks.0": {
          "boundary": "tie",
          "detail": "blocks.0: same-shape weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.1": {
          "boundary": "tie",
          "detail": "blocks.1: same-shape weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.2": {
          "boundary": "tie",
          "detail": "blocks.2: same-shape weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.3": {
          "boundary": "tie",
          "detail": "blocks.3: same-shape weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.0+blocks.1": {
          "boundary": "tie",
          "detail": "blocks.0~blocks.1: tied weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.0+blocks.2": {
          "boundary": "tie",
          "detail": "blocks.0~blocks.2: tied weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.0+blocks.3": {
          "boundary": "tie",
          "detail": "blocks.0~blocks.3: tied weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.1+blocks.2": {
          "boundary": "tie",
          "detail": "blocks.1~blocks.2: tied weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.1+blocks.3": {
          "boundary": "tie",
          "detail": "blocks.1~blocks.3: tied weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "weight_tie:blocks.2+blocks.3": {
          "boundary": "tie",
          "detail": "blocks.2~blocks.3: tied weights",
          "status": "declined",
          "reason": "no_tied_values"
        },
        "kv_latent_share:blocks.0": {
          "boundary": "intra",
          "detail": "blocks.0: shared-data kv projections",
          "status": "declined",
          "reason": "no certified common factor"
        },
        "kv_latent_share:blocks.1": {
          "status": "declined",
          "reason": "error",
          "error": "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 2359296 bytes. Error code 12 (Cannot allocate memory)"
        },
        "kv_latent_share:blocks.2": {
          "boundary": "intra",
          "detail": "blocks.2: shared-data kv projections",
          "status": "declined",
          "reason": "no certified common factor"
        },
        "kv_latent_share:blocks.3": {
          "boundary": "intra",
          "detail": "blocks.3: shared-data kv projections",
          "status": "declined",
          "reason": "no certified common factor"
        }
      },
      "morphism_fires": {},
      "n_rewritten": 0,
      "blocks": {},
      "in_place": false,
      "shared_params": true,
      "end_to_end": {
        "error": "RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 8192000 bytes. Error code 12 (Cannot allocate memory)"
      },
      "wall_time_s": 43.53308725357056
    },
    "outcome": "completed"
  },
  "delivery": {
    "laws": [
      "residual_reassoc",
      "out_in_compose",
      "residual_absorb",
      "norm_cascade",
      "weight_tie",
      "kv_latent_share"
    ],
    "optimize_rest": false,
    "window_excluded": true,
    "window_exclusion_reason": "window match declined under the guarded run",
    "seconds": 4.9,
    "n_rewritten": 14,
    "morphism_fires": {
      "norm_cascade": 12,
      "weight_tie": 1
    },
    "end_to_end": {
      "max_abs_diff": 1.4781951904296875e-05,
      "max_rel_diff": 1.0847653102289108e-06
    },
    "in_place": false,
    "per_law": {
      "out_in_compose": {
        "matched": 10,
        "grafted": 0,
        "declined": 10,
        "skipped": 0,
        "reasons": {
          "no_improvement": 10
        }
      },
      "norm_cascade": {
        "matched": 12,
        "grafted": 12,
        "declined": 0,
        "skipped": 0,
        "reasons": {}
      },
      "weight_tie": {
        "matched": 79,
        "grafted": 1,
        "declined": 0,
        "skipped": 78,
        "reasons": {
          "node already rewritten": 78
        }
      },
      "kv_latent_share": {
        "matched": 12,
        "grafted": 0,
        "declined": 0,
        "skipped": 12,
        "reasons": {
          "node already rewritten": 12
        }
      }
    },
    "matches": {
      "out_in_compose:blocks.0+blocks.1": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.0.out_proj \u2218 blocks.1.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.1+blocks.2": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.1.out_proj \u2218 blocks.2.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.2+blocks.3": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.2.out_proj \u2218 blocks.3.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.3+blocks.4": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.3.out_proj \u2218 blocks.4.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.4+blocks.5": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.4.out_proj \u2218 blocks.5.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.5+blocks.6": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.5.out_proj \u2218 blocks.6.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.6+blocks.7": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.6.out_proj \u2218 blocks.7.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.7+blocks.8": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.7.out_proj \u2218 blocks.8.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.8+blocks.9": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.8.out_proj \u2218 blocks.9.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "out_in_compose:blocks.9+blocks.10": {
        "status": "declined",
        "reason": "no_improvement",
        "boundary": "chain",
        "detail": "blocks.9.out_proj \u2218 blocks.10.in_projs",
        "cost_before": 1840123904.0,
        "cost_after": 1840123904.0
      },
      "norm_cascade:blocks.0": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.0 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 2.5213162493074435e-07
      },
      "norm_cascade:blocks.1": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.1 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 3.526764924367132e-07
      },
      "norm_cascade:blocks.2": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.2 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 1.5307266576774107e-07
      },
      "norm_cascade:blocks.3": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.3 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 1.0003477203431209e-07
      },
      "norm_cascade:blocks.4": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.4 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 7.714223203917101e-08
      },
      "norm_cascade:blocks.5": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.5 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 6.88405265583707e-08
      },
      "norm_cascade:blocks.6": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.6 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 1.1106595659707318e-07
      },
      "norm_cascade:blocks.7": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.7 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 1.004562925456669e-07
      },
      "norm_cascade:blocks.8": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.8 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 9.319301445253111e-08
      },
      "norm_cascade:blocks.9": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.9 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 1.4189854383033375e-07
      },
      "norm_cascade:blocks.10": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.10 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 1.4490870857287386e-07
      },
      "norm_cascade:blocks.11": {
        "status": "grafted",
        "boundary": "intra",
        "detail": "blocks.11 pre-norm gain folds into in_projs",
        "cost_before": 920064000.0,
        "cost_after": 919965696.0,
        "rel_diff": 1.3278309785702136e-07
      },
      "weight_tie:blocks.0": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:emb+head": {
        "status": "grafted",
        "boundary": "tie",
        "detail": "emb~head: tied weights",
        "rel_diff": 0.0,
        "tied": [
          [
            "emb__p_weight",
            "head__p_weight"
          ]
        ]
      },
      "weight_tie:blocks.0+blocks.1": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.0+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.1+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.2+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.3+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.4+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5+blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5+blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.5+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.6+blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.6+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.6+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.6+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.6+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.7+blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.7+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.7+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.7+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.8+blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.8+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.8+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.9+blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.9+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "weight_tie:blocks.10+blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.0": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.1": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.10": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.11": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.2": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.3": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.4": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.5": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.6": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.7": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.8": {
        "status": "skipped",
        "reason": "node already rewritten"
      },
      "kv_latent_share:blocks.9": {
        "status": "skipped",
        "reason": "node already rewritten"
      }
    },
    "blocks": {}
  },
  "verify_fresh": {
    "passed": true,
    "max_rel": 1.158577516358039e-06
  },
  "timing": {
    "eager": 235.963,
    "morphism": 218.202,
    "speedup": 1.0814
  }
}
```

</details>

