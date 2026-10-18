//! `mlopt-experiments` — the kit this repo's experiments run on.
//!
//! The smallest honest assembly of the modular handl-lang stack for the
//! paper's measurements — the same component set catopt-kit composes
//! (`packages/catopt-kit`, in-tree in this repo, is the precedent; this
//! module is the post-ADR-0128 spelling over the renamed `hnl-engine-*`
//! crates — the same edges catopt-kit's manifest now declares):
//!
//! 1. **Language** — the canonical `Language` assembled from data via
//!    [`handl_kit_data_canonical::language_from_yaml`].
//! 2. **Install** — [`handl_plugin_lib_handl::install_default_language`]
//!    makes it the process-wide language that lazily-built `.hnl` module
//!    bridge runtimes bind (ADR-0128 §"the 0.8 seam").
//! 3. **Engines** — `data` lex, `pratt` parse, `tree-walk` eval: the
//!    smallest set the measurements need. The [`EngineRegistry`]
//!    selects by name, so more can register without touching this crate.
//! 4. **Modules** — `host-tensor`'s `tensor.*` op surface (18 ops)
//!    composed as a [`HandlModule`] over `tensor_op_registry()`, under
//!    the canonical fp host functions.
//! 5. **Imports** — a [`ModuleLoader`] wired into the eval engine, so
//!    `.hnl` sources can `import "./f.hnl"` and — under
//!    [`Kit::new_with_project`] — `from "host-tensor" import tensor`
//!    resolves through the project's `.handl.toml` `[dependencies]`.
//!
//! Values cross the engine boundary as `serde_json::Value` (the
//! `{"__tensor__": …}` wire form); inside the tree-walk engine they are
//! [`TaggedValue`]s — a tensor op result is `TaggedValue::Tensor`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use handl_host_tensor::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};
use handl_language::{LangError, Language, Source, Value};
use handl_platform::ModuleLookup;
use handl_plugin_lib_handl::HandlModule;
use handl_plugin_module::ModuleRegistry;
use handl_runtime::{EngineRegistry, ModuleLoader, Runtime, RuntimeConfig};
use handl_value::TaggedValue;
use hnl_engine_exec::{EvalEngine, EvalRequest, ImportResolver};
use hnl_engine_exec_tree_walk::TreeWalkEngine;
use hnl_engine_lex_pattern::DataLexerEngine;
use hnl_engine_parse_pratt::PrattEngine;

/// The engine registry for the experiments' runtime — the same three
/// engines the canonical kit binds as its default pipeline, and nothing
/// else.
///
/// Exposed so a consumer can register additional engines (e.g.
/// `"bytecode"`) before passing the registry to
/// [`Kit::new_with_registry`].
pub fn engine_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::new();
    registry.register_lex("data", Box::new(DataLexerEngine::new()));
    registry.register_parse("pratt", PrattEngine::new());
    registry.register_eval("tree-walk", Box::new(TreeWalkEngine::new()));
    registry
}

/// The `tensor` module the experiments compose — `host-tensor`'s
/// `index.hnl` interface backed by the registered Rust impls.
pub fn tensor_module() -> HandlModule {
    HandlModule::new("tensor", TENSOR_HNL, &[]).with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()))
}

/// The module chain wired into every eval engine: the `tensor`
/// [`ModuleRegistry`] layered under the canonical fp host functions
/// (`field_access` is what makes `tensor.matmul(a, b)` dispatch —
/// `tensor` is unbound → the string `"tensor"`, `field_access`
/// concatenates to `"tensor.matmul"`, and `call` routes the qualified
/// name through [`ModuleLookup`] → the op's Rust impl).
pub fn module_chain() -> Rc<dyn ModuleLookup> {
    let modules: Rc<dyn ModuleLookup> = Rc::new(ModuleRegistry::new().with(tensor_module()));
    let chain: Rc<dyn ModuleLookup> = handl_runtime::fp::canonical_fp_host_functions(Some(modules));
    chain
}

/// The assembled handl kit: the canonical `Language` bound to the
/// `data`/`pratt`/`tree-walk` pipeline with the `tensor.*` host op
/// surface and the file/name import machinery wired.
///
/// Construct via [`Kit::new`] (no project context — `import` resolves
/// relative to the working directory) or [`Kit::new_with_project`] (a
/// `.handl.toml` manifest anchors `[dependencies]` name resolution).
///
/// The kit is **not** `Send`: the tree-walk engine keeps `Rc`-shared
/// state — build one per thread.
pub struct Kit {
    /// The assembled runtime (lex → parse → eval with desugar at parse).
    runtime: Runtime,
    /// A second `tree-walk` engine wired identically to the runtime's —
    /// kept for [`Kit::run_tagged`], the `TaggedValue`-returning eval
    /// path that skips the `serde_json::Value` materialization.
    tagged_engine: Box<dyn EvalEngine>,
}

impl Kit {
    /// Assemble the kit with no project context (relative `import`
    /// resolves against the process working directory).
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
    /// register. The config still selects `data`/`pratt`/`tree-walk` by
    /// name, so the registry must provide at least those three.
    ///
    /// # Errors
    ///
    /// As [`Kit::new`].
    pub fn new_with_registry(registry: EngineRegistry) -> Result<Self, LangError> {
        Self::build(None, HashMap::new(), registry)
    }

    /// Assemble the kit for a project directory: reads
    /// `<project_root>/.handl.toml` (if present) and wires its
    /// `[dependencies]`/`[patch]` tables into the module loader, with
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

    /// Like [`Kit::run`], but returns the engine's internal
    /// [`TaggedValue`] — a tensor op result comes back as
    /// `TaggedValue::Tensor`, not the `Obj` fallback a wire decode
    /// would produce.
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

    /// The underlying [`Runtime`].
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// The [`Language`] this kit binds.
    pub fn lang(&self) -> &Language {
        self.runtime.lang()
    }
}
