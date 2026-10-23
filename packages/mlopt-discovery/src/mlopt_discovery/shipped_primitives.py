"""Shipped machine-minted primitives — the ``synop_*`` fold family.

The ``overnight.py --corpus zoo-prim`` arm minted 24 primitive ops
for manual subgraph shapes recurring across >=2 held-out zoo modules
and denoted by no bank op; 21 cleared the full admission gauntlet
(``docs/reports/results_overnight_prim_v2.json`` — every stage green,
all ``machine-original``).  The minted objects were runtime-only —
the ``synop_*`` op registration, the torch denotation and the
``prim:synop_*`` fold law all died with the driver process.  This
module is their shipped home: the records embed verbatim below and
materialize at call time into the same objects the mint built.

What ships per primitive (all derivable from the embedded record):

* the **op** — :func:`declare` registers ``synop_<hash>`` with
  :func:`mlopt_core.ir.op_def` (the mint's name IS the honest one:
  ``sha256(alpha-normal pattern)[:10]`` — a content hash, not a
  coined label);
* the **denotation** — :func:`register` seats a torch binding for
  the op on the ambient ``mlopt_torch.torch_bridge._IR_TO_TORCH``
  table, so every ``OpTable.full()``-backed ``TorchSink`` (existing
  instances included — the table IS the ambient dict) can lower it;
  the binding evaluates the spelled pattern itself — the op's
  semantics ARE the manual subgraph (ADR 0007: the executor table
  is the meaning), so the fold is sound by construction and the
  mint's fp64 site checks all measured ``equal``;
* the **fold law** — ``prim:synop_<hash>`` rewrites the pattern to
  the ``synop_<hash>`` call; :func:`folds` materializes the 21;
* the **definitional premise** — ``synop_<hash>.expand``
  (``synop_call -> pattern``), the axiom the cert leg's anchored
  derivation unions in when the bare bank cannot derive the fold;
  :func:`premises` materializes them and each fold carries its
  premise as ``rule.premises`` — the same convention
  ``pipeline.Proposal.as_rule`` mints.

Home choice (recorded honestly)
-------------------------------

The records could ship in ``mlopt_core.laws.minted`` — the
``conv_bn_fold`` channel — but the substrate is torch-free: the
*denotation* (the executor-table binding that is the op's meaning)
lives in ``mlopt_torch`` and the mint machinery in
``mlopt_discovery``.  Seating the ops in ``mlopt_core`` would
register generator entries whose bindings this repo pins on anyway,
and pushing the folds into core ``DEFAULT`` would change extraction
for every downstream consumer — including torch-free and
non-torch-sink contexts where a ``synop_*`` term can never be
lowered.  So the family ships domain-side, exactly like
:mod:`mlopt_discovery.machine_pack`: a distinct opt-in set deployed
through the ``rules=`` seam, never folded into ``DEFAULT``.  No
eqsat change; no pin bump.

Opt-in contract
---------------

::

    from mlopt_discovery import shipped_primitives as sp
    from mlopt_orchestrator.optimize import search, lower
    from mlopt_torch.adapters import TorchSink, TorchSource

    sp.register()                       # op_def + torch bindings
    res = search(m, x, source=TorchSource(), rules=sp.prim_default())
    out = lower(res, x, sink=TorchSink(), verify=True)

``register()`` is the deliverability half — without it saturation
still fires the folds (pattern matching needs no op table) but
extraction's ``supported_ops`` bound correctly refuses a form no
sink can lower.  ``prim_default()`` unions the fold set onto the
orchestrator's composed ``DEFAULT_RULES`` (lazy import — the fold
materialization itself stays torch-free).  The ``expand`` premises
are *not* saturation rules — they are the cert-leg's axioms; adding
them to a search rule set would re-add the unfolded spelling.

Provenance
----------

:data:`MACHINE_MINTED` is the graphopt-side provenance ledger — the
fold (``prim:synop_*``), expand (``synop_*.expand``) and op
(``synop_*``) names this module ships, all ``prov=machine``.  Core's
``mlopt_core.laws.provenance.law_provenance`` is registry-level over
the *core* library: it does not see this set (an unshipped-to-core
name classifies ``"human"`` by default).  :func:`law_provenance`
here is the honest classifier for a name drawn from either library.
Each record also carries the mint's own measurements — ``mods``
(the bound modules), ``sites``/``equal_sites`` (the fp64 denotation
check the arm ran) and ``handler`` (the grammar minter's verdict —
``"minted"`` where a handler body verified, ``"refused"`` where the
primitive ships on its definitional denotation alone).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Const, Op, op_def
from mlopt_core.laws import tags
from mlopt_core.laws.ruleset import RuleSet
from mlopt_core.laws.serialize import object_from_data

__all__ = [
    "FOLD_LAW",
    "MACHINE_MINTED",
    "SYNOP_SHIPPED",
    "declare",
    "expand_name",
    "folds",
    "law_provenance",
    "premises",
    "prim_default",
    "records",
    "register",
    "ruleset",
    "spec",
    "synop_name",
]

#: The verbatim ``law_to_data`` records of the 21 admitted folds,
#: keyed by the minted ``synop_*`` op name, plus each primitive's
#: mint-side measurements (``alpha`` — the content the op name
#: hashes; ``mods`` — the bound-module set the k=2 gate measured;
#: ``sites``/``equal_sites`` — the denotation's fp64 site check;
#: ``handler`` — the grammar minter's verdict).  Frozen artifacts of
#: the ``zoo-prim`` mint at ``docs/reports/results_overnight_prim_v2.json``
#: (protocol ``seed=0``, ``xbind=2``): regenerate by re-running the
#: pool mint — the op names are content hashes, so a deterministic
#: re-mint reproduces these byte-identically.
_RECORDS: dict[str, Any] = json.loads(
    r"""{
 "synop_03889d8fe1": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), 'v4')), 'v5')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "mvar": "v484"
             },
             {
              "mvar": "v485"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v307"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_03889d8fe1",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v484"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v307"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_03889d8fe1"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v484",
     "v485",
     "v307",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          "v484",
          "v485"
         ]
        ],
        "v307"
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         "v484",
         "v485"
        ]
       ],
       "v307"
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_2b8cfd1c90": {
  "alpha": "(add (mul 'v0', (add 'v1', (mul 'v2', 'v3'))), (mul 'v4', 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "mvar": "v267"
         },
         {
          "args": [
           {
            "mvar": "v347"
           },
           {
            "mvar": "v348"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_2b8cfd1c90",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v267"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v348"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_2b8cfd1c90"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v267",
     "v347",
     "v348",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        "v267",
        [
         "mul",
         "v347",
         "v348"
        ]
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       "v267",
       [
        "mul",
        "v347",
        "v348"
       ]
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_39f76b66f8": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), 'v4')), 'v5')), 'v6')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v509"
               },
               {
                "mvar": "v510"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "mvar": "v485"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v307"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_39f76b66f8",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v509"
     },
     {
      "mvar": "v510"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v307"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_39f76b66f8"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v509",
     "v510",
     "v485",
     "v307",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          [
           "mul",
           "v509",
           "v510"
          ],
          "v485"
         ]
        ],
        "v307"
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         [
          "mul",
          "v509",
          "v510"
         ],
         "v485"
        ]
       ],
       "v307"
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_4c5296b3ba": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), 'v3')), (mul 'v4', (unsqueeze 'v5', dim=a0)))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "mvar": "v346"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v268"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "args": [
         {
          "mvar": "v355"
         }
        ],
        "attrs": {
         "dim": "a97"
        },
        "op": "unsqueeze"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_4c5296b3ba",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v268"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v355"
     }
    ],
    "attrs": {
     "a97": "a97"
    },
    "op": "synop_4c5296b3ba"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_4c657cb2c0": {
  "alpha": "(add 'v0', (mul (unsqueeze 'v1', dim=-1), (unsqueeze (select (linear 'v2', 'v3', 'v4'), dim=1, index=a1), dim=a0)))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v13"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v419"
         }
        ],
        "attrs": {
         "dim": -1
        },
        "op": "unsqueeze"
       },
       {
        "args": [
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v633"
             },
             {
              "mvar": "v634"
             },
             {
              "mvar": "v635"
             }
            ],
            "attrs": {},
            "op": "linear"
           }
          ],
          "attrs": {
           "dim": 1,
           "index": "a155"
          },
          "op": "select"
         }
        ],
        "attrs": {
         "dim": "a75"
        },
        "op": "unsqueeze"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_4c657cb2c0",
   "rhs": {
    "args": [
     {
      "mvar": "v13"
     },
     {
      "mvar": "v419"
     },
     {
      "mvar": "v633"
     },
     {
      "mvar": "v634"
     },
     {
      "mvar": "v635"
     }
    ],
    "attrs": {
     "a155": "a155",
     "a75": "a75"
    },
    "op": "synop_4c657cb2c0"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_4f8fefdd38": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), 'v4')), (mul 'v5', 'v6'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "args": [
             {
              "mvar": "v362"
             },
             {
              "mvar": "v363"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v268"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_4f8fefdd38",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v362"
     },
     {
      "mvar": "v363"
     },
     {
      "mvar": "v268"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_4f8fefdd38"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v345",
     "v362",
     "v363",
     "v268",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        [
         "mul",
         "v345",
         [
          "add",
          "v362",
          "v363"
         ]
        ],
        "v268"
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       [
        "mul",
        "v345",
        [
         "add",
         "v362",
         "v363"
        ]
       ],
       "v268"
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_5147826430": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), (mul 'v4', 'v5'))), 'v6')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "mvar": "v492"
             },
             {
              "mvar": "v493"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v480"
           },
           {
            "mvar": "v481"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_5147826430",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v492"
     },
     {
      "mvar": "v493"
     },
     {
      "mvar": "v480"
     },
     {
      "mvar": "v481"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_5147826430"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v492",
     "v493",
     "v480",
     "v481",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          "v492",
          "v493"
         ]
        ],
        [
         "mul",
         "v480",
         "v481"
        ]
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         "v492",
         "v493"
        ]
       ],
       [
        "mul",
        "v480",
        "v481"
       ]
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_571fb768b0": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add 'v2', (mul 'v3', 'v4'))), (mul 'v5', 'v6')))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "mvar": "v749"
           },
           {
            "args": [
             {
              "mvar": "v771"
             },
             {
              "mvar": "v772"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v739"
         },
         {
          "mvar": "v740"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_571fb768b0",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v749"
     },
     {
      "mvar": "v771"
     },
     {
      "mvar": "v772"
     },
     {
      "mvar": "v739"
     },
     {
      "mvar": "v740"
     }
    ],
    "attrs": {},
    "op": "synop_571fb768b0"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v749",
     "v771",
     "v772",
     "v739",
     "v740"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         "v749",
         [
          "mul",
          "v771",
          "v772"
         ]
        ]
       ],
       [
        "mul",
        "v739",
        "v740"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        "v749",
        [
         "mul",
         "v771",
         "v772"
        ]
       ]
      ],
      [
       "mul",
       "v739",
       "v740"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_5e58f26875": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), (mul 'v3', 'v4'))), (mul 'v5', 'v6'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "mvar": "v346"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v358"
           },
           {
            "mvar": "v359"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_5e58f26875",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v358"
     },
     {
      "mvar": "v359"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_5e58f26875"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v345",
     "v346",
     "v358",
     "v359",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        [
         "mul",
         "v345",
         "v346"
        ],
        [
         "mul",
         "v358",
         "v359"
        ]
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       [
        "mul",
        "v345",
        "v346"
       ],
       [
        "mul",
        "v358",
        "v359"
       ]
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_629f9ef13b": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), (mul 'v4', 'v5'))), 'v6'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v791"
             },
             {
              "mvar": "v792"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "args": [
             {
              "mvar": "v801"
             },
             {
              "mvar": "v802"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v138"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_629f9ef13b",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v791"
     },
     {
      "mvar": "v792"
     },
     {
      "mvar": "v801"
     },
     {
      "mvar": "v802"
     },
     {
      "mvar": "v138"
     }
    ],
    "attrs": {},
    "op": "synop_629f9ef13b"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v791",
     "v792",
     "v801",
     "v802",
     "v138"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         [
          "mul",
          "v791",
          "v792"
         ],
         [
          "mul",
          "v801",
          "v802"
         ]
        ]
       ],
       "v138"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        [
         "mul",
         "v791",
         "v792"
        ],
        [
         "mul",
         "v801",
         "v802"
        ]
       ]
      ],
      "v138"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_62d36aefdb": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), 'v4')), (mul 'v5', 'v6')))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v769"
             },
             {
              "mvar": "v770"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "mvar": "v750"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v739"
         },
         {
          "mvar": "v740"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_62d36aefdb",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v769"
     },
     {
      "mvar": "v770"
     },
     {
      "mvar": "v750"
     },
     {
      "mvar": "v739"
     },
     {
      "mvar": "v740"
     }
    ],
    "attrs": {},
    "op": "synop_62d36aefdb"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v769",
     "v770",
     "v750",
     "v739",
     "v740"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         [
          "mul",
          "v769",
          "v770"
         ],
         "v750"
        ]
       ],
       [
        "mul",
        "v739",
        "v740"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        [
         "mul",
         "v769",
         "v770"
        ],
        "v750"
       ]
      ],
      [
       "mul",
       "v739",
       "v740"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_6c709f774a": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), (mul 'v3', 'v4'))), 'v5')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "mvar": "v473"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v480"
           },
           {
            "mvar": "v481"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_6c709f774a",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v473"
     },
     {
      "mvar": "v480"
     },
     {
      "mvar": "v481"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_6c709f774a"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v473",
     "v480",
     "v481",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         "v473"
        ],
        [
         "mul",
         "v480",
         "v481"
        ]
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        "v473"
       ],
       [
        "mul",
        "v480",
        "v481"
       ]
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_889d740672": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add 'v2', (mul 'v3', 'v4'))), 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "mvar": "v743"
           },
           {
            "args": [
             {
              "mvar": "v793"
             },
             {
              "mvar": "v794"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v138"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_889d740672",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v743"
     },
     {
      "mvar": "v793"
     },
     {
      "mvar": "v794"
     },
     {
      "mvar": "v138"
     }
    ],
    "attrs": {},
    "op": "synop_889d740672"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v743",
     "v793",
     "v794",
     "v138"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         "v743",
         [
          "mul",
          "v793",
          "v794"
         ]
        ]
       ],
       "v138"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        "v743",
        [
         "mul",
         "v793",
         "v794"
        ]
       ]
      ],
      "v138"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_93e57c595e": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), 'v4')), 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v791"
             },
             {
              "mvar": "v792"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "mvar": "v744"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v138"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_93e57c595e",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v791"
     },
     {
      "mvar": "v792"
     },
     {
      "mvar": "v744"
     },
     {
      "mvar": "v138"
     }
    ],
    "attrs": {},
    "op": "synop_93e57c595e"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v791",
     "v792",
     "v744",
     "v138"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         [
          "mul",
          "v791",
          "v792"
         ],
         "v744"
        ]
       ],
       "v138"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        [
         "mul",
         "v791",
         "v792"
        ],
        "v744"
       ]
      ],
      "v138"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_976f9836d7": {
  "alpha": "(add (mul 'v0', (add 'v1', (mul 'v2', 'v3'))), (mul 'v4', (unsqueeze 'v5', dim=a0)))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "mvar": "v267"
         },
         {
          "args": [
           {
            "mvar": "v347"
           },
           {
            "mvar": "v348"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "args": [
         {
          "mvar": "v370"
         }
        ],
        "attrs": {
         "dim": "a101"
        },
        "op": "unsqueeze"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_976f9836d7",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v267"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v348"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v370"
     }
    ],
    "attrs": {
     "a101": "a101"
    },
    "op": "synop_976f9836d7"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_98a002d64b": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), (mul 'v3', (unsqueeze 'v4', dim=a0)))), 'v5')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "mvar": "v473"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v480"
           },
           {
            "args": [
             {
              "mvar": "v500"
             }
            ],
            "attrs": {
             "dim": "a128"
            },
            "op": "unsqueeze"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_98a002d64b",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v473"
     },
     {
      "mvar": "v480"
     },
     {
      "mvar": "v500"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {
     "a128": "a128"
    },
    "op": "synop_98a002d64b"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_b246dcb838": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), (mul 'v4', 'v5')))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "mvar": "v749"
           },
           {
            "mvar": "v750"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v739"
         },
         {
          "mvar": "v740"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_b246dcb838",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v749"
     },
     {
      "mvar": "v750"
     },
     {
      "mvar": "v739"
     },
     {
      "mvar": "v740"
     }
    ],
    "attrs": {},
    "op": "synop_b246dcb838"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v749",
     "v750",
     "v739",
     "v740"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         "v749",
         "v750"
        ]
       ],
       [
        "mul",
        "v739",
        "v740"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        "v749",
        "v750"
       ]
      ],
      [
       "mul",
       "v739",
       "v740"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_bb8c54c201": {
  "alpha": "(add (mul 'v0', 'v1'), (mul 'v2', 'v3'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "mvar": "v224"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_bb8c54c201",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v224"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_bb8c54c201"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v224",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       "v224"
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      "v224"
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        4,
        16
       ],
       "v216": [
        4,
        16
       ],
       "v223": [
        4,
        16
       ],
       "v224": [
        4,
        16
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v224": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v224": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v224": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "BlockSparseAttention",
   "DecoupledMLA",
   "DilatedBlockAttention",
   "GRUFusedGate",
   "GatedDeltaRule",
   "GatedLinearAttention",
   "HighwayGate",
   "MambaSelectiveScan",
   "PartialRotary",
   "RWKVMixBlock",
   "RealGatedLRU",
   "RotaryPair",
   "SinkWindowAttention",
   "SlidingWindowAttention"
  ],
  "sites": 16
 },
 "synop_d5b128282a": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', (mul 'v3', 'v4'))), 'v5')), 'v6')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "mvar": "v484"
             },
             {
              "args": [
               {
                "mvar": "v511"
               },
               {
                "mvar": "v512"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v307"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_d5b128282a",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v484"
     },
     {
      "mvar": "v511"
     },
     {
      "mvar": "v512"
     },
     {
      "mvar": "v307"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_d5b128282a"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v484",
     "v511",
     "v512",
     "v307",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          "v484",
          [
           "mul",
           "v511",
           "v512"
          ]
         ]
        ],
        "v307"
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         "v484",
         [
          "mul",
          "v511",
          "v512"
         ]
        ]
       ],
       "v307"
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_f08a53f9a0": {
  "alpha": "(add (mul 'v0', (add 'v1', (mul 'v2', (unsqueeze 'v3', dim=a0)))), (mul 'v4', 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "mvar": "v267"
         },
         {
          "args": [
           {
            "mvar": "v347"
           },
           {
            "args": [
             {
              "mvar": "v381"
             }
            ],
            "attrs": {
             "dim": "a103"
            },
            "op": "unsqueeze"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_f08a53f9a0",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v267"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v381"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {
     "a103": "a103"
    },
    "op": "synop_f08a53f9a0"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_f3290a1163": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), 'v3')), (mul 'v4', 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "mvar": "v346"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v268"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_f3290a1163",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v268"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_f3290a1163"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v345",
     "v346",
     "v268",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        [
         "mul",
         "v345",
         "v346"
        ],
        "v268"
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       [
        "mul",
        "v345",
        "v346"
       ],
       "v268"
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 }
}"""
)

#: The minted op names this module ships (``synop_<sha256[:10]>``).
SYNOP_SHIPPED: frozenset[str] = frozenset(_RECORDS)

#: Every machine-minted name this module ships — fold laws
#: (``prim:synop_*``), definitional premises (``synop_*.expand``) and
#: the ops themselves.  The domain-side mirror of core's
#: ``MACHINE_SHIPPED``: these never enter ``DEFAULT`` or the core
#: registry, so core's ``law_provenance`` does not classify them —
#: this set is the ledger for the domain-side shipped machines.
MACHINE_MINTED: frozenset[str] = frozenset(
    [f"prim:{n}" for n in _RECORDS]
    + [f"{n}.expand" for n in _RECORDS]
    + list(_RECORDS)
)

#: The tag the fold rules carry — a structural fold collapsing a
#: consumer pattern into one kernel, same class as ``conv_bn_fold``.
FOLD_LAW = (
    "machine-minted primitive fold — the manual subgraph collapses "
    "to the {synop} primitive (zoo-prim mint; admitted by the full "
    "gauntlet, machine-original; denotation definitional — the op's "
    "torch binding IS the spelled subgraph)"
)

#: Law text for the definitional premise — the cert leg's anchor.
_EXPAND_LAW = (
    "{synop} unfolds to its mined denotation — the definitional "
    "premise the cert leg's anchored derivation unions in when the "
    "bare bank cannot derive the fold"
)

_MATERIALIZED: dict[str, tuple[Rewrite, Rewrite]] = {}


def synop_name(fold_name: str) -> str:
    """Return the op a ``prim:synop_*`` fold name mints (identity otherwise)."""
    return fold_name[5:] if fold_name.startswith("prim:") else fold_name


def expand_name(synop: str) -> str:
    """Return the definitional-premise law name for a ``synop_*`` op."""
    return f"{synop_name(synop)}.expand"


def records() -> dict[str, Any]:
    """Return the embedded records — the frozen mint artifacts."""
    return _RECORDS


def spec(synop: str) -> Any:
    """Return the op's denotation as a genkernel spec tuple.

    Rendered from the fold record's LHS with metavar names kept —
    the ``_pat_spec`` convention the mint used: leaf metavars stay
    their pattern names (the call's positional args bind them) and
    attr metavars stay their names inside the attr dicts (the call's
    attrs carry the bound values back in at eval time).
    """
    return _spec_of(_materialize(synop_name(synop))[0].lhs)


def _spec_of(pat: Any) -> Any:
    """Render an ``Op`` pattern as a genkernel spec — metavar names kept."""
    if isinstance(pat, Op):
        parts = [_spec_of(a) for a in pat.args]
        if pat.attrs:
            parts.append(dict(pat.attrs))
        return (pat.op, *parts)
    if isinstance(pat, Const):
        return pat.value
    return pat  # metavar leaf — the name IS the binding key


def _spec_bind(spec_: Any, attrs: dict, attr_mvars: tuple) -> Any:
    """Instantiate a spec's attr-metavar values under the call's attrs.

    The minted call carries its attr metavars as real op attrs; the
    spelled denotation needs those values back inside the spec's attr
    dicts — a dict value equal to an attr-mvar name resolves through
    the call's attrs, everything else passes through literal.
    """
    if isinstance(spec_, (tuple, list)):
        out = []
        for e in spec_:
            if isinstance(e, dict):
                out.append(
                    {
                        k: (
                            attrs.get(v, v)
                            if isinstance(v, str) and v in attr_mvars
                            else v
                        )
                        for k, v in e.items()
                    }
                )
            else:
                out.append(_spec_bind(e, attrs, attr_mvars))
        return tuple(out) if isinstance(spec_, tuple) else out
    return spec_


def _binding(spec_: Any, leaf_mvars: tuple, attr_mvars: tuple) -> Any:
    """Return the op's torch binding — its denotation, spelled out.

    ``fn(*args, **attrs) = eval(spec, {mvar: arg}, ambient)`` — the
    op IS the manual subgraph: equal to the fold's LHS at every bound
    site by construction (the mint measured ``equal`` on every site —
    ``sites``/``equal_sites`` in the record).  The ambient table is
    looked up per call so post-hoc overrides keep reaching it.
    """

    def fn(*args: Any, **attrs: Any) -> Any:
        import mlopt_torch.torch_bridge as tb

        from mlopt_discovery.genkernel import _eval_spec

        env = dict(zip(leaf_mvars, args, strict=True))
        return _eval_spec(
            _spec_bind(spec_, attrs, attr_mvars),
            env,
            tb._IR_TO_TORCH,
        )

    return fn


def _materialize(synop: str) -> tuple[Rewrite, Rewrite]:
    """Build ``(fold, expand)`` for one record — memoized.

    The fold rebuilds through the shipped record codec
    (:func:`~mlopt_core.laws.serialize.object_from_data`) — the same
    object the mint admitted — then gets the shipped integration
    fields (``law`` text, ``FUSION`` tag) and its premise attached,
    exactly as ``pipeline.Proposal.as_rule`` minted it.
    """
    if synop in _MATERIALIZED:
        return _MATERIALIZED[synop]
    rec = _RECORDS[synop]
    fold = object_from_data(rec["fold"])
    expand = Rewrite(
        expand_name(synop),
        fold.rhs,
        fold.lhs,
        law=_EXPAND_LAW.format(synop=synop),
        tags=frozenset({tags.SIMPLIFICATION}),
    )
    out = Rewrite(
        name=fold.name,
        lhs=fold.lhs,
        rhs=fold.rhs,
        law=FOLD_LAW.format(synop=synop),
        tags=frozenset({tags.FUSION}),
        error_bound=fold.error_bound,
        bound_norm=fold.bound_norm,
        derivation=fold.derivation,
        cond=fold.cond,
        dspec=fold.dspec,
    )
    object.__setattr__(out, "premises", (expand,))
    _MATERIALIZED[synop] = (out, expand)
    op_def(
        synop,
        len(fold.rhs.args),
        1,
        law=(
            f"synthesized primitive — denotation is the mined manual "
            f"subgraph {rec['alpha']}"
        ),
    )
    return _MATERIALIZED[synop]


def declare() -> None:
    """Register every shipped ``synop_*`` op — torch-free.

    Writes the :class:`~mlopt_core.ir.GenDef` entries
    (:func:`mlopt_core.ir.op_def`) so ``generator(name)`` resolves
    and the ops are first-class IR citizens.  Idempotent;
    materializing the folds declares them anyway — this is the
    explicit spelling for callers that want the ops without the
    laws' materialization or the torch bindings.
    """
    for name in _RECORDS:
        _materialize(name)


def register() -> int:
    """Install the ops' torch denotations; return the op count.

    ``declare()`` plus one ambient binding per op:
    ``_IR_TO_TORCH[synop] = fn`` where ``fn`` replays the spelled
    subgraph through the same ambient table — so nested ``synop``
    denotations and post-hoc overrides compose.  Seating on the
    ambient dict reaches every ``OpTable.full()``-backed sink,
    including already-constructed ``TorchSink`` instances
    (``supported_ops`` is the table's live key set).  Idempotent —
    re-registration rewrites byte-identical bindings.
    """
    import mlopt_torch.torch_bridge as tb

    declare()
    for name in _RECORDS:
        fold, _expand = _materialize(name)
        tb._IR_TO_TORCH[name] = _binding(
            _spec_of(fold.lhs),
            tuple(fold.rhs.args),
            tuple(fold.rhs.attrs),
        )
    return len(_RECORDS)


def folds() -> tuple[Rewrite, ...]:
    """Return the 21 shipped fold laws — ``pattern -> synop_*(args)``."""
    return tuple(_materialize(n)[0] for n in _RECORDS)


def premises() -> tuple[Rewrite, ...]:
    """Return the definitional ``synop_*.expand`` premises — cert axioms.

    NOT a saturation set: expanding ``synop -> pattern`` inside a
    search would re-add the unfolded spelling the fold collapses.
    These are the anchored-derivation axioms a cert/verify leg unions
    into its axiom set.
    """
    return tuple(_materialize(n)[1] for n in _RECORDS)


def ruleset(name: str = "prim_folds") -> RuleSet:
    """Return the fold set as a named :class:`RuleSet` for ``rules=``."""
    return RuleSet(
        name,
        folds(),
        description=(
            "the shipped machine-minted primitive folds — "
            f"{len(_RECORDS)} synop_* ops from the zoo-prim mint "
            "(docs/reports/results_overnight_prim_v2.json); "
            "denotations register via shipped_primitives.register()"
        ),
    )


def prim_default(*, base: Any = None) -> RuleSet:
    """Return ``DEFAULT_RULES + prim folds`` — the deployed opt-in set.

    *base* is the ruleset the folds union onto — ``None`` resolves
    the orchestrator's composed ``DEFAULT_RULES`` (imported lazily so
    materialization stays torch-free).  The folds stay a distinct
    set, never folded into ``DEFAULT`` — the provenance audit reads
    the boundary, and a consumer without the torch bindings pays only
    the saturation enodes (extraction's ``supported_ops`` bound never
    selects an unlowerable form).
    """
    if base is None:
        from mlopt_orchestrator.optimize import default_rules

        base = default_rules()
    return base + ruleset()


def law_provenance(rule: object) -> Literal["machine", "human"]:
    """Classify a rewrite's authorship across BOTH shipped libraries.

    :func:`mlopt_core.laws.provenance.law_provenance` is
    registry-level over the core library — a name shipped only
    domain-side (the ``prim:synop_*`` folds, their ``synop_*.expand``
    premises and the ``synop_*`` ops) classifies ``"human"`` there by
    default.  This classifier unions the two ledgers: this module's
    :data:`MACHINE_MINTED` names report ``"machine"``; anything else
    delegates to the core registry.
    """
    from mlopt_core.laws.provenance import (
        law_provenance as _core_provenance,
    )

    name = (
        rule if isinstance(rule, str) else getattr(rule, "name", rule)
    )
    if name in MACHINE_MINTED:
        return "machine"
    return _core_provenance(name)
