# Retro — spike Q1: can a tensor cross the boundary?

**Date**: 2026-10-14.  **Plan**: 0029, **Q1**.
**Status**: **Q1a answered (mechanism exists); Q1b (a runnable demo) is
the remaining measurement.**

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

## Q1b — the remaining measurement

A runnable demonstration: register a host function that receives a
`Resource` and returns one (a stand-in for `matmul` over a host-owned
buffer), run a small term through handl's exec, and check the numeric
result against the same computation done directly.

What it would prove: that the *channel* works end-to-end through exec
(not merely that the variant exists).  What it would still not prove:
that the host object is a *torch* tensor — that is a PyO3 question, and
it belongs to Q3 (integration direction), not here.

Not run yet: the host box was busy with a full-suite benchmark (see
`GOALS.md` G1); the Rust target is warm, so it is a small increment.
