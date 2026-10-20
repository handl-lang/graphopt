"""Shipped minted objects — stored-composite materialization in core.

The discovery pipeline's stored objects (``mlopt_discovery.evidence``)
are `Rewrite`-shaped: a record carries the pattern pair, ``cond`` /
``dspec`` / ``derivation`` data, the ``serializable`` / ``missing_hooks``
honesty flags, and — for a synthesized composite — a ``"guard"`` field:
the ``guard_pats`` re-expression maps plus the two parent names from
which ``mlopt_core.meta._compose_guards`` rebuilds the composed
``check`` / ``derive`` hooks at admit time.  That record IS the object —
the hooks are code, but the *spec* of the guard is data.

A minted object earns the library only through the admission gauntlet
(``evidence.run_gauntlet`` — every stage green, ``usable``).  Shipping it
then means materializing the *same measured object* at import: the
records are embedded verbatim below (``_RECORDS``), and
:func:`materialize` replays the store's ``_rebuild_stored`` semantics
over the embedded table — each ``syn_*`` node rebuilds its pattern from
its record and its guard from the named parents, resolved first against
the embedded records (recursively, so a composite-of-composites
rebuilds) and then against the shipped library by name.

Two deliberate differences from the store path, both louder:

* a parent that resolves short of its declared strength — absent from
  both tables, or a record whose ``missing_hooks`` exceed what the
  transport restored — **raises** rather than degrading to the bare
  pattern: a shipped object may never silently lose its guard;
* the leaf parents are the *library objects themselves*
  (``BATCH_NORM_RULES`` / ``CONV_FOLD_RULES`` members), not records —
  a live ``Rewrite`` is its own full-strength declaration, and the
  composed hooks close over exactly the hooks it was minted against.

The shipped composite is therefore the first library member whose
``missing_hooks`` is non-empty — ``("check", "derive")`` — by class, not
by accident: its guard is the transported premise composition, carried
inspectably as ``guard_pats`` + ``parents`` on the object.  The
serialization census (``tests/test_law_serialize.py``) records this
*transported* class separately from the full-data majority.

The embedded records below are the verbatim ``law_json`` rows
``store_object`` wrote for the conv+BN fold's premise closure of minted
intermediates (``bench/drivers/probes/conv_fold_premise_pay_probe.py`` mint;
``project/retros/minted-composite-default.md`` records the admission
account).  They are frozen artifacts: the composite's guard names them
by minted name, and they are scaffolding — they do not ship as rules.
"""

from __future__ import annotations

import json
from typing import Any

from mlopt_core import meta
from mlopt_core.egraph import Rewrite
from mlopt_core.laws.serialize import object_from_data
from mlopt_core.rulecache import _dec_binding

__all__ = ["materialize"]

#: The verbatim ``law_json`` records of the minted intermediates the
#: shipped composites descend from, keyed by minted (``syn_*``) name.
#: Leaf parents are NOT here — they resolve against the shipped library
#: by name at materialize time.
_RECORDS: dict[str, Any] = json.loads(
    r"""{
 "syn_batch_norm_expand__conv_channel_shift_sub_sp_6": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "batch_norm_expand",
    "conv_channel_shift_sub_sp"
   ],
   "pat1": [
    [
     "$attr:TR",
     {
      "a": false
     }
    ],
    [
     "$attr:M",
     {
      "a": 0.1
     }
    ],
    [
     "$attr:EP",
     {
      "a": 1e-05
     }
    ],
    [
     "$attr:CU",
     {
      "a": false
     }
    ],
    [
     "x",
     {
      "t": {
       "args": [
        {
         "mvar": "v0"
        },
        {
         "mvar": "v1"
        },
        {
         "mvar": "v2"
        }
       ],
       "attrs": {
        "padding": {
         "__tuple__": [
          1,
          1
         ]
        },
        "stride": {
         "__tuple__": [
          1,
          1
         ]
        }
       },
       "op": "conv2d"
      }
     }
    ],
    [
     "w",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "rm",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "rv",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "cv",
     {
      "t": {
       "args": [
        {
         "mvar": "v5"
        }
       ],
       "attrs": {
        "shape": "@1:CH"
       },
       "op": "reshape"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: batch_norm_expand then conv_channel_shift_sub_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_batch_norm_expand__conv_channel_shift_sub_sp_6",
  "rhs": {
   "args": [
    {
     "args": [
      {
       "args": [
        {
         "mvar": "v3"
        }
       ],
       "attrs": {
        "shape": "@1:CH"
       },
       "op": "reshape"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v0"
          },
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          }
         ],
         "attrs": {
          "padding": {
           "__tuple__": [
            1,
            1
           ]
          },
          "stride": {
           "__tuple__": [
            1,
            1
           ]
          }
         },
         "op": "conv2d"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "mvar": "v6"
              }
             ],
             "attrs": {
              "shape": "@1:CH"
             },
             "op": "reshape"
            },
            {
             "mvar": "@1:EPS"
            }
           ],
           "attrs": {},
           "op": "add"
          }
         ],
         "attrs": {},
         "op": "rsqrt"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     ],
     "attrs": {},
     "op": "mul"
    },
    {
     "args": [
      {
       "mvar": "v4"
      }
     ],
     "attrs": {
      "shape": "@1:CH"
     },
     "op": "reshape"
    }
   ],
   "attrs": {},
   "op": "add"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 },
 "syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "syn_batch_norm_expand__conv_channel_shift_sub_sp_6",
    "conv_channel_scale_sp"
   ],
   "pat1": [
    [
     "v0",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "v1",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "v2",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "v3",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "v4",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "v5",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "v6",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "args": [
        {
         "mvar": "v2"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v5"
            }
           ],
           "attrs": {
            "shape": "@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "sub"
      }
     }
    ],
    [
     "sv",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v6"
            }
           ],
           "attrs": {
            "shape": "@1:@1:CH"
           },
           "op": "reshape"
          },
          {
           "mvar": "@1:@1:EPS"
          }
         ],
         "attrs": {},
         "op": "add"
        }
       ],
       "attrs": {},
       "op": "rsqrt"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: syn_batch_norm_expand__conv_channel_shift_sub_sp_6 then conv_channel_scale_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48",
  "rhs": {
   "args": [
    {
     "args": [
      {
       "args": [
        {
         "mvar": "v3"
        }
       ],
       "attrs": {
        "shape": "@1:@1:CH"
       },
       "op": "reshape"
      },
      {
       "args": [
        {
         "mvar": "v0"
        },
        {
         "args": [
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@2:WV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        }
       ],
       "attrs": {
        "padding": {
         "__tuple__": [
          1,
          1
         ]
        },
        "stride": {
         "__tuple__": [
          1,
          1
         ]
        }
       },
       "op": "conv2d"
      }
     ],
     "attrs": {},
     "op": "mul"
    },
    {
     "args": [
      {
       "mvar": "v4"
      }
     ],
     "attrs": {
      "shape": "@1:@1:CH"
     },
     "op": "reshape"
    }
   ],
   "attrs": {},
   "op": "add"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 },
 "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48",
    "conv_channel_scale_l_sp"
   ],
   "pat1": [
    [
     "v0",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "v1",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "v2",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "v3",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "v4",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "v5",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "v6",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "sv",
     {
      "t": {
       "args": [
        {
         "mvar": "v3"
        }
       ],
       "attrs": {
        "shape": "@1:@1:@1:CH"
       },
       "op": "reshape"
      }
     }
    ],
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "args": [
        {
         "mvar": "v1"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v6"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:CH"
               },
               "op": "reshape"
              },
              {
               "mvar": "@1:@1:@1:EPS"
              }
             ],
             "attrs": {},
             "op": "add"
            }
           ],
           "attrs": {},
           "op": "rsqrt"
          }
         ],
         "attrs": {
          "shape": "@1:@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "mvar": "v2"
          },
          {
           "args": [
            {
             "args": [
              {
               "mvar": "v5"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@1:CH"
             },
             "op": "reshape"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "sub"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v6"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:CH"
               },
               "op": "reshape"
              },
              {
               "mvar": "@1:@1:@1:EPS"
              }
             ],
             "attrs": {},
             "op": "add"
            }
           ],
           "attrs": {},
           "op": "rsqrt"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48 then conv_channel_scale_l_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209",
  "rhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@2:WV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      },
      {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "args": [
      {
       "mvar": "v4"
      }
     ],
     "attrs": {
      "shape": "@1:@1:@1:CH"
     },
     "op": "reshape"
    }
   ],
   "attrs": {},
   "op": "add"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 },
 "syn_syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209__conv_channel_shift_sp_896": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209",
    "conv_channel_shift_sp"
   ],
   "pat1": [
    [
     "v0",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "v1",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "v2",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "v3",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "v4",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "v5",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "v6",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:WV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ],
    [
     "cv",
     {
      "t": {
       "args": [
        {
         "mvar": "v4"
        }
       ],
       "attrs": {
        "shape": "@1:@1:@1:@1:CH"
       },
       "op": "reshape"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209 then conv_channel_shift_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209__conv_channel_shift_sp_896",
  "rhs": {
   "args": [
    {
     "mvar": "v0"
    },
    {
     "args": [
      {
       "args": [
        {
         "mvar": "v1"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v6"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:@1:CH"
               },
               "op": "reshape"
              },
              {
               "mvar": "@1:@1:@1:@1:EPS"
              }
             ],
             "attrs": {},
             "op": "add"
            }
           ],
           "attrs": {},
           "op": "rsqrt"
          }
         ],
         "attrs": {
          "shape": "@1:@1:@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v3"
          }
         ],
         "attrs": {
          "shape": "@1:@1:@1:@1:CH"
         },
         "op": "reshape"
        }
       ],
       "attrs": {
        "shape": "@1:@2:WV"
       },
       "op": "reshape"
      }
     ],
     "attrs": {},
     "op": "mul"
    },
    {
     "args": [
      {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v4"
          }
         ],
         "attrs": {
          "shape": "@1:@1:@1:@1:CH"
         },
         "op": "reshape"
        }
       ],
       "attrs": {
        "shape": "@2:SV"
       },
       "op": "reshape"
      }
     ],
     "attrs": {},
     "op": "add"
    }
   ],
   "attrs": {
    "padding": {
     "__tuple__": [
      1,
      1
     ]
    },
    "stride": {
     "__tuple__": [
      1,
      1
     ]
    }
   },
   "op": "conv2d"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 }
}"""
)


def _resolve(name: str, lib: dict[str, Rewrite], seen: frozenset[str]):
    """Rebuild *name*'s Rewrite; return ``(rule, transported)``.

    *transported* names the hooks the ``"guard"`` spec restored —
    ``()`` for a library-resolved leaf (a live object is its own
    full-strength declaration) or an unguarded record.  Raises
    ``ValueError`` on any shortfall the store path would degrade
    through: shipping a weakened composite is a smuggle, not a
    partial admit.
    """
    rec = _RECORDS.get(name)
    if rec is None:
        try:
            return lib[name], ()
        except KeyError:
            raise ValueError(
                f"minted object {name!r} names a parent that is "
                f"neither an embedded record nor a library rule"
            ) from None
    if name in seen:
        raise ValueError(f"cyclic guard spec at {name!r}")
    rule = object_from_data(rec)
    guard = rec.get("guard")
    if not isinstance(guard, dict):
        return rule, ()
    parents = tuple(guard.get("parents") or ())
    if len(parents) != 2:
        raise ValueError(
            f"minted object {name!r} carries a guard spec with "
            f"{len(parents)} parents — composition is binary"
        )
    try:
        pats = (
            _dec_binding(guard["pat1"]),
            _dec_binding(guard["pat2"]),
        )
    except Exception as exc:
        raise ValueError(
            f"minted object {name!r} carries an undecodable guard spec"
        ) from exc
    resolved = [
        _resolve_parent(pname, lib, seen | {name}) for pname in parents
    ]
    return _compose_record(name, rule, resolved, parents, pats)


def _resolve_parent(
    pname: str, lib: dict[str, Rewrite], seen: frozenset[str]
) -> Rewrite:
    """Resolve one guard parent — record first (recursive), then lib.

    A record-resolved parent must rebuild at *full declared strength*
    (``missing_hooks`` minus transported empty) — the same bar
    ``evidence._rebuild_stored`` holds, raised here into an error
    rather than a silent degrade.
    """
    prule, ptrans = _resolve(pname, lib, seen)
    prec = _RECORDS.get(pname)
    if prec is not None and set(prec.get("missing_hooks") or ()) - set(
        ptrans
    ):
        raise ValueError(
            f"minted parent {pname!r} rebuilt weaker than it "
            f"declared (missing "
            f"{sorted(set(prec['missing_hooks']) - set(ptrans))}) "
            f"— composing over it would smuggle a weakened guard"
        )
    return prule


def _compose_record(
    name: str,
    rule: Rewrite,
    resolved: list[Rewrite],
    parents: tuple[str, ...],
    pats: tuple[Any, Any],
):
    """Re-run ``_compose_guards`` over resolved parents; rewrap *rule*.

    Returns ``(out, transported)`` — *transported* names the hooks the
    guard spec restored.  Raises ``ValueError`` when the spec yields no
    hooks (a guard-less rebuild is not the declared object).
    """
    check, drv = meta._compose_guards(
        resolved[0], resolved[1], pats[0], pats[1]
    )
    if check is None and drv is None:
        raise ValueError(
            f"minted object {name!r} declares a guard spec but "
            f"transports no hooks"
        )
    # Same policy as synthesize_rules.offer / rulecache._dec_rule: the
    # composite derive attaches only where the RHS carries the "@i:"
    # placeholders it refills.
    if not (
        drv is not None and meta._rhs_derive_placeholders(rule.rhs)
    ):
        drv = None
    out = Rewrite(
        name=rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        law=rule.law,
        tags=rule.tags,
        error_bound=rule.error_bound,
        bound_norm=rule.bound_norm,
        derivation=rule.derivation,
        cond=rule.cond,
        dspec=rule.dspec,
        check=check,
        derive=drv,
    )
    object.__setattr__(out, "parents", parents)
    object.__setattr__(out, "guard_pats", pats)
    meta.SYNTH_PARENTS[out.name] = (parents[0], parents[1])
    transported = tuple(
        h
        for h, f in (("check", check), ("derive", drv))
        if f is not None
    )
    return out, transported


def materialize(
    name: str,
    lib: dict[str, Rewrite],
    *,
    ship_as: str | None = None,
    tags: Any = None,
    law: str | None = None,
) -> Rewrite:
    """Materialize an embedded minted record as a shipped ``Rewrite``.

    *name* is the minted record's own (``syn_*``) name; *lib* is the
    name→rule table the leaf parents resolve against (the shipped
    library families — e.g. ``BATCH_NORM_RULES`` + ``CONV_FOLD_RULES``).
    *ship_as* renames the root object for the library (promotion
    practice: minted ``syn_*`` names ship under their law name, as the
    promoted-laws table did); *tags* / *law* override the record's
    fields for the shipped object — integration decisions, recorded
    here rather than smuggled into the verbatim record.  The
    intermediate nodes keep their minted names: they are scaffolding,
    not shipped rules.

    The returned object carries ``parents`` / ``guard_pats`` — the
    same provenance ``meta.synthesize_rules`` mints carry — so the
    store's ``_guard_transport_spec`` re-encodes it faithfully on a
    re-store, and ``missing_hooks`` on it honestly reports the
    procedural hooks its class carries.
    """
    rule, _ = _resolve(name, lib, frozenset())
    if ship_as is not None:
        parents = getattr(rule, "parents", None)
        object.__setattr__(rule, "name", ship_as)
        if parents is not None:
            meta.SYNTH_PARENTS[ship_as] = parents
    if tags is not None:
        object.__setattr__(rule, "tags", frozenset(tags))
    if law is not None:
        object.__setattr__(rule, "law", law)
    return rule
