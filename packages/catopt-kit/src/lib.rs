//! `catopt-kit` — catopt's handl composition seam.
//!
//! The smallest honest assembly of the modular handl-lang stack for
//! catopt's use:
//!
//! 1. **Language** — the canonical `Language` assembled from data via
//!    [`handl_kit_data_canonical::language_from_yaml`] (tokens, grammar,
//!    eval schemes, AST shapes — all YAML, no engine-specific code).
//! 2. **Install** — [`handl_plugin_lib_handl::install_default_language`]
//!    makes it the process-wide language that lazily-built `.hnl` module
//!    bridge runtimes bind (ADR-0128 §"the 0.8 seam").
//! 3. **Engines** — the smallest set catopt needs: the data lexer
//!    (`"data"`), the Pratt parser (`"pratt"`), and the tree-walk
//!    evaluator (`"tree-walk"`). No sdkgen/compile/lint/lsp — the
//!    [`EngineRegistry`] selects engines by name, so a later consumer
//!    can register more without touching this crate.
//! 4. **Modules** — `host-tensor`'s `tensor.*` op surface (18 ops)
//!    composed as a [`HandlModule`] over `tensor_op_registry()`, under
//!    the canonical fp host functions (`field_access` is what makes
//!    `tensor.matmul(a, b)` dispatch: `tensor` is unbound → the string
//!    `"tensor"`, `field_access` concatenates to `"tensor.matmul"`, and
//!    `call` routes the qualified name through [`ModuleLookup`] → the
//!    `OpFunction` → the Rust impl).
//! 5. **Imports** — a [`ModuleLoader`] (the Phase 9.5 file/dependency
//!    resolver) is wired into the eval engine, so `.hnl` sources can
//!    `import "./f.hnl"` and — under [`CatoptKit::new_with_project`] —
//!    `from "host-tensor" import tensor` resolves through the project's
//!    `.handl.toml` `[dependencies]` table.
//!
//! Values cross the engine boundary as `serde_json::Value` (the
//! `{"__tensor__": …}` wire form); inside the tree-walk engine they are
//! [`TaggedValue`]s — a tensor op result is [`TaggedValue::Tensor`], not
//! a plain object. [`CatoptKit::run_tagged`] exposes that path.

#![warn(missing_docs)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use handl_host_tensor::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};
use handl_language::{LangError, Language, Source, Value};
use handl_platform::ModuleLookup;
use handl_plugin_engine_exec::{EvalEngine, EvalRequest, ImportResolver};
use handl_plugin_engine_exec_tree_walk::TreeWalkEngine;
use handl_plugin_engine_lex_pattern::DataLexerEngine;
use handl_plugin_engine_parse_pratt::PrattEngine;
use handl_plugin_lib_handl::HandlModule;
use handl_plugin_module::ModuleRegistry;
use handl_runtime::{EngineRegistry, ModuleLoader, Runtime, RuntimeConfig};
use handl_value::TaggedValue;

/// The engine registry for catopt's runtime — the same three engines the
/// canonical kit binds as its default pipeline, and nothing else.
///
/// Exposed so a downstream consumer can register additional engines
/// (e.g. `"bytecode"`) before passing the registry to
/// [`CatoptKit::new_with_registry`].
pub fn engine_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::new();
    registry.register_lex("data", Box::new(DataLexerEngine::new()));
    registry.register_parse("pratt", PrattEngine::new());
    registry.register_eval("tree-walk", Box::new(TreeWalkEngine::new()));
    registry
}

/// The `tensor` module catopt composes — `host-tensor`'s `index.hnl`
/// interface (18 `op` declarations) backed by the registered Rust impls.
///
/// `HandlModule::new` runs the declaration-stripping prep on the source
/// (`cap`/`op`/`handl` lines are interface text, not evaluable handl);
/// `with_ops` exposes each op under its local name (`"tensor.matmul"` →
/// `matmul`) dispatching into the shared [`OpRegistry`] impl table.
pub fn tensor_module() -> HandlModule {
    HandlModule::new("tensor", TENSOR_HNL, &[]).with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()))
}

/// The module chain [`CatoptKit`] wires into every eval engine:
/// the `tensor` [`ModuleRegistry`] layered under the canonical fp host
/// functions (`fn.fanout_fn`, `field_access`).
///
/// `field_access` is the piece that makes the ambient `tensor.f` call
/// form work with no globals injected: `tensor` is an unbound
/// identifier (→ the string `"tensor"`), `field_access("tensor", "f")`
/// concatenates to `"tensor.f"`, and the `call` scheme dispatches the
/// qualified name through this chain (the same module-path construction
/// `tag.tag` uses in the canonical kit).
pub fn module_chain() -> Rc<dyn ModuleLookup> {
    let modules: Rc<dyn ModuleLookup> = Rc::new(ModuleRegistry::new().with(tensor_module()));
    let chain: Rc<dyn ModuleLookup> = handl_runtime::fp::canonical_fp_host_functions(Some(modules));
    chain
}

/// catopt's assembled handl kit: the canonical `Language` bound to the
/// `data`/`pratt`/`tree-walk` pipeline with the `tensor.*` host op
/// surface and the file/name import machinery wired.
///
/// Construct via [`CatoptKit::new`] (no project context — `import`
/// resolves relative to the working directory) or
/// [`CatoptKit::new_with_project`] (a `.handl.toml` manifest anchors
/// `[dependencies]` name resolution, the `from "pkg" import …` seam).
///
/// The kit is **not** `Send`: the tree-walk engine keeps `Rc`-shared
/// state — build one per thread, as the upstream `Runtime` prescribes.
pub struct CatoptKit {
    /// The assembled runtime (lex → parse → eval with desugar at parse).
    runtime: Runtime,
    /// A second `tree-walk` engine wired identically to the runtime's —
    /// kept for [`CatoptKit::run_tagged`], the `TaggedValue`-returning
    /// eval path that skips the `serde_json::Value` materialization the
    /// `Function`-boundary `eval` performs.
    tagged_engine: Box<dyn EvalEngine>,
}

impl CatoptKit {
    /// Assemble the kit with no project context (no `.handl.toml`
    /// dependencies; relative `import "./f.hnl"` resolves against the
    /// process working directory).
    ///
    /// # Errors
    ///
    /// Returns [`LangError`] if the canonical language fails to assemble
    /// or an engine name fails to resolve.
    pub fn new() -> Result<Self, LangError> {
        Self::build(None, HashMap::new(), engine_registry())
    }

    /// Assemble the kit over a caller-supplied [`EngineRegistry`] — the
    /// injection seam for engines this crate does not statically
    /// register (ADR-0128 §0.5). The config still selects
    /// `data`/`pratt`/`tree-walk` by name, so the registry must provide
    /// at least those three.
    ///
    /// # Errors
    ///
    /// As [`CatoptKit::new`].
    pub fn new_with_registry(registry: EngineRegistry) -> Result<Self, LangError> {
        Self::build(None, HashMap::new(), registry)
    }

    /// Assemble the kit for a project directory: reads
    /// `<project_root>/.handl.toml` (if present) and wires its
    /// `[dependencies]` table into the module loader, with
    /// `project_root` as the module root for relative imports.
    ///
    /// Each `[dependencies]` entry maps an import name to a package
    /// directory: `from "host-tensor" import tensor` resolves
    /// `index.hnl` inside the named directory.
    ///
    /// # Errors
    ///
    /// Returns [`LangError`] if the manifest exists but fails to parse,
    /// or if assembly fails.
    pub fn new_with_project(project_root: PathBuf) -> Result<Self, LangError> {
        let deps = handl_platform::resolve_dependencies(&project_root)
            .map_err(|e| LangError::engine(format!("project dependencies: {e}")))?;
        Self::build(Some(project_root), deps, engine_registry())
    }

    /// The shared assembly step.
    ///
    /// Mirrors `kit-canonical`'s `Kit::build_kit` minus what catopt does
    /// not compose: no stdlib/meta module chain, no `self`/`hello`
    /// globals, no typecheck engine. (`self` is the `Language`-value
    /// binding meta functions need — drop it until a catopt `.hnl`
    /// module calls `eval`/`compile` on the language itself.)
    fn build(
        module_root: Option<PathBuf>,
        dependencies: HashMap<String, PathBuf>,
        registry: EngineRegistry,
    ) -> Result<Self, LangError> {
        let lang = handl_kit_data_canonical::language_from_yaml()?;
        // Install as the process-wide default language for `.hnl` module
        // bridge runtimes (ADR-0128 §0.8). First install wins — a
        // rejection just means another host installed first, so it is
        // not an error here.
        let _ = handl_plugin_lib_handl::install_default_language(lang.clone());

        let registry = Rc::new(registry);
        let config = RuntimeConfig::new()
            .with_lex("data")
            .with_parse("pratt")
            .with_eval("tree-walk");

        let chain = module_chain();

        // The `ModuleLoader` for `import "path"` / `from "pkg" import n`.
        // Two-phase construction (upstream docs): create the loader, then
        // late-bind an eval engine that resolves nested imports through
        // the loader itself.
        let lang_rc = Rc::new(lang.clone());
        let loader = Rc::new(
            ModuleLoader::new(
                Rc::clone(&lang_rc),
                registry
                    .lex_engine("data")
                    .ok_or_else(|| LangError::engine("missing lex engine 'data'"))?
                    .box_clone(),
                registry
                    .parse_engine("pratt")
                    .ok_or_else(|| LangError::engine("missing parse engine 'pratt'"))?
                    .box_clone(),
                module_root.clone(),
            )
            .with_dependencies(dependencies),
        );
        let mut loader_eval = TreeWalkEngine::new()
            .with_modules(Rc::clone(&chain))
            .with_import_resolver(Rc::clone(&loader) as Rc<dyn ImportResolver>);
        // Ambient globals the language data declares (`true`/`false`,
        // …) must be in scope inside imported modules too.
        for (name, val) in &lang.globals {
            loader_eval = loader_eval.with_global(name, val.clone());
        }
        loader.set_eval_engine(Box::new(loader_eval));
        let import_resolver: Rc<dyn ImportResolver> = loader;

        // A tagged-eval twin of the runtime's wiring for `run_tagged` —
        // modules, ambient globals, and the same import resolver.
        let mut tagged_engine = registry
            .eval_engine("tree-walk")
            .ok_or_else(|| LangError::engine("missing eval engine 'tree-walk'"))?
            .with_modules(Rc::clone(&chain));
        for (name, val) in &lang.globals {
            tagged_engine = tagged_engine.with_global(name, val.clone());
        }
        tagged_engine = tagged_engine.with_import_resolver(Rc::clone(&import_resolver));

        let runtime = Runtime::from_registry_with_resolver(
            lang,
            &registry,
            &config,
            Some(chain),
            None,
            Some(import_resolver),
            module_root,
        )?;
        Ok(Self {
            runtime,
            tagged_engine,
        })
    }

    /// `lex → parse → eval` — run a `.hnl` snippet, returning the
    /// `serde_json::Value` boundary form (`{"__tensor__": …}` for a
    /// tensor).
    ///
    /// # Errors
    ///
    /// Propagates the first stage error.
    pub fn run(&self, src: &str) -> Result<Value, LangError> {
        self.runtime.run(&Source::new(src))
    }

    /// Like [`CatoptKit::run`], but returns the engine's internal
    /// [`TaggedValue`] — the shape values take inside the tree-walk
    /// engine. A tensor op result comes back as
    /// [`TaggedValue::Tensor`], the real tensor value (not the
    /// `Obj` fallback a `{"__tensor__": …}` map would decode to on a
    /// kernel without the Tensor variant).
    ///
    /// # Errors
    ///
    /// Propagates the first stage error.
    pub fn run_tagged(&self, src: &str) -> Result<TaggedValue, LangError> {
        let toks = self.runtime.lex(&Source::new(src))?;
        let ast = self.runtime.parse(&toks)?;
        self.tagged_engine
            .eval_tagged(self.runtime.lang(), &EvalRequest::new(ast))
    }

    /// The underlying [`Runtime`] (staged `lex`/`parse`/`eval` over
    /// [`Source`]/`Tokens`/`Ast`).
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// The [`Language`] this kit binds.
    pub fn lang(&self) -> &Language {
        self.runtime.lang()
    }
}
