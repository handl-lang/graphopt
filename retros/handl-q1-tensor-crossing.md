---
title: 'Retro — spike Q1: can a tensor cross the boundary?'
type: retro
date: 2026-10-14
---

# Retro — spike Q1: can a tensor cross the boundary?

**Plan**: 0029, **Q1**.
host-function seam) are both measured.**  What is *not* done is a
runnable demonstration; see "Q1b" for why that is now off the critical
path.

## The question

ADR 0010 claims catopt's tensor primitives can be handl **host
functions**.  That claim is only as good as the value channel: a host
function must be able to *receive and return a tensor*.  Reading the
docs is not enough — the type has to be able to carry one.

## Q1a — the value model (measured by reading the types)

handl's value stack:

| layer | type | can it carry a tensor? |
|---|---|---|
| language-level | `handl_language::Value` = **`serde_json::Value`** | ❌ JSON only (null/bool/number/string/array/object) |
| runtime | `handl_value::TaggedValue` | ⚠️ richer, and **yes — with a caveat** |

`TaggedValue`'s variants: `Null`, `Bool`, `Int`, `Float`, `Sym`, `Str`,
**`Bytes(Rc<Vec<u8>>)`**, `Arr`, `Obj`, `Pair`, `EitherLeft/Right`,
`Arrow`, and —

```rust
Resource(Rc<ForeignResource>)

pub struct ForeignResource {
    pub ptr: *mut c_void,                            // opaque C handle
    pub finalizer: Option<unsafe fn(*mut c_void)>,   // cleanup on drop
    dropped: AtomicBool,                             // double-free guard
}
```

`ForeignResource`'s own docstring is explicit: *"`ptr` must be a valid
opaque handle obtained from a C function (e.g. a file descriptor cast to
a pointer, an `mmap` region, a socket handle).  This type never
dereferences `ptr` — it only passes it to `finalizer`."*

**So the answer to Q1 is yes — a tensor crosses by reference, with no
copy**: the host allocates it, hands handl a pointer with a finalizer,
and the value carries the pointer.  Nothing serializes, nothing copies.

### The caveats, which are the honest part

1. **The type is by convention, not by type system.**  There is no
   typed opaque variant (`Arc<dyn Any>`-style) — a `Resource` is a raw
   `*mut c_void`, so the host must know what it points at.  A tensor and
   a file descriptor are the same shape to handl.
2. **Unsafe.**  The host upholds validity and the finalizer contract;
   the double-free guard is handl's only protection.
3. **Single-threaded.**  `Rc`-held, explicitly not `Send`/`Sync` — fine
   for a search loop, wrong for parallel execution.
4. **The FFI seam is thin today.**  `vm_primitives`'s `ffi` host
   function *"handles a few native arithmetic ops (`add`, `mul`, `sub`,
   `div`); other dispatches return `Null`"*.  A tensor op would need a
   registered host function/module (`handl_platform::{Function, Module}`,
   with `NativeOpCtx::import_resolver` as the seam), not the built-in
   `ffi`.
5. **`Value` vs `TaggedValue`.**  The *language-level* value is
   `serde_json::Value`; the resource lives in the *runtime* value.  So a
   tensor-bearing value is a runtime notion — the language's `eval`
   schemes never see it as JSON.

## Q1b — the host-function seam (measured by reading it and its tests)

The seam a tensor op would register into:

```rust
impl Function for AddHostFn {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext)
        -> Result<Value, handl_platform::EvalError>
}
```

with `Value = serde_json::Value` (the same alias as the language-level
value).  That is the finding: **the host-function seam is JSON-valued**,
while the resource lives in the runtime's richer `TaggedValue`.

A `Resource` does have a JSON encoding — but read what it says about
itself (`crates/handl-value/src/lib.rs`):

```rust
/// JSON encoding is `{"__resource__": {"ptr": <address>}}` — a debug
TaggedValue::Resource(r) => json!({"__resource__": {"ptr": r.ptr as i64}}),
```

and, at the decoder:

> `{"__resource__": {"ptr": <n>}}` — a raw process-local pointer; in
> another run it decodes to a **dangling** [`ForeignResource`].

So a tensor reaches a host function by one of three routes:

| route | what it costs | honest verdict |
|---|---|---|
| **smuggle the pointer** through `{"__resource__": {"ptr": n}}` | nothing (no copy) | **valid in-process** — the same address space — and **dangling across processes**; the encoding is documented as debug-only and the boundary is unsafe |
| **copy the bytes** — `Bytes` (`{"__bytes__": [n, …]}`) plus an `Obj` for shape/dtype | a full copy per crossing | safe and portable; wrong for a hot loop |
| **extend the seam** to speak `TaggedValue` | a handl change | the clean fix; not ours to make unilaterally |

### Why this is off the critical path

ADR 0011 decides that **rewriting** is canonical in the calculus and
**execution** stays catopt's — so in the first version **no tensor needs
to cross at all**.  The useful integration is the language: catopt's
fragments in, handl's `Language` + rewrite/typecheck engines, terms out.
Q1b's demo would measure a channel that v1 does not use.

That also means the unsafe, single-threaded, process-local caveats above
do not block the integration — they would only matter if tensors were
made to cross, which is a *later* question with a named answer (the
pointer route, valid in-process, which is exactly what the Q3-(A) PyO3
peer is).

## What is still not measured

- **A tensor op running through exec end-to-end.**  The channel analysis
  above says it can (in-process, via the pointer encoding), but no run
  demonstrates it.  Deliberately deferred, with the reason recorded.
- **Whether the pointer route survives PyO3.**  A PyO3 extension owns
  Python objects; handing handl a `*mut c_void` to a Python object with a
  finalizer that decrefs is the natural shape — but it is a *design*, not
  a measurement, and it belongs to the peer's build (Q3-(A) follow-up).
