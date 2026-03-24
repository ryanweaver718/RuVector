# Model Swap Targets: all-MiniLM-L6-v2 References

All files referencing `all-MiniLM-L6-v2`, hardcoded `384` dimension, or `DEFAULT_MODEL` for embeddings.

Legend:
- **CHANGE** = Default/hardcoded value that must be updated for a model swap
- **REF** = Reference, comment, or documentation (update for accuracy but not functional)
- **TEST** = Test code that will need updating

---

## 1. npm/packages/ruvector/src/ (TypeScript/JS Source)

### core/onnx-embedder.ts
- L4: `Provides real transformer-based embeddings using all-MiniLM-L6-v2` — REF (comment)
- L10: `384-dimensional semantic embeddings` — REF (comment)
- L93: `const DEFAULT_MODEL = 'all-MiniLM-L6-v2';` — **CHANGE**
- L149: `config.modelId || DEFAULT_MODEL` — CHANGE (uses DEFAULT_MODEL)
- L212: `config.modelId || DEFAULT_MODEL` — CHANGE (uses DEFAULT_MODEL)
- L306: `parallelEmbedder.dimension || 384` — **CHANGE** (fallback dimension)
- L378: `embedder ? embedder.dimension() : 384` — **CHANGE** (fallback dimension)
- L402: `dimension: embedder ? embedder.dimension() : 384` — **CHANGE** (fallback dimension)
- L403: `model: DEFAULT_MODEL` — CHANGE (uses DEFAULT_MODEL)

### core/onnx-embedder.js (compiled JS — mirrors .ts)
- L5: comment — REF
- L11: comment — REF
- L99: `const DEFAULT_MODEL = 'all-MiniLM-L6-v2';` — **CHANGE**
- L153: `config.modelId || DEFAULT_MODEL` — CHANGE
- L209: `config.modelId || DEFAULT_MODEL` — CHANGE
- L291: `parallelEmbedder.dimension || 384` — **CHANGE**
- L350: `embedder ? embedder.dimension() : 384` — **CHANGE**
- L364: `dimension: embedder ? embedder.dimension() : 384` — **CHANGE**
- L365: `model: DEFAULT_MODEL` — CHANGE

### core/onnx-embedder.d.ts
- L4: comment — REF
- L10: comment — REF

### core/onnx-optimized.ts
- L31: `Model to use (default: 'all-MiniLM-L6-v2')` — REF (JSDoc)
- L63-68: Model registry entry `'all-MiniLM-L6-v2': { onnx, fp16, int8, tokenizer URLs }` — **CHANGE** (add new model or change default)
- L69: `dimension: 384` — **CHANGE**
- L77: `dimension: 384` (all-mpnet) — REF (different model's dimension)
- L84: `dimension: 384` (bge-small) — REF (different model's dimension)
- L185: `private dimension = 384;` — **CHANGE** (default dimension)
- L189: `modelId: config.modelId ?? 'all-MiniLM-L6-v2'` — **CHANGE**

### core/onnx-optimized.js (compiled JS — mirrors .ts)
- L65-70: Model registry `'all-MiniLM-L6-v2'` — **CHANGE**
- L71: `dimension: 384` — **CHANGE**
- L79: `dimension: 384` (all-mpnet) — REF
- L86: `dimension: 384` (bge-small) — REF
- L169: `this.dimension = 384` — **CHANGE**
- L171: `modelId: config.modelId ?? 'all-MiniLM-L6-v2'` — **CHANGE**

### core/onnx-optimized.d.ts
- L18: `Model to use (default: 'all-MiniLM-L6-v2')` — REF (type comment)

### core/intelligence-engine.ts
- L79: comment `default: 256, 384 for ONNX` — REF
- L179: comment `use 384 dimensions (MiniLM default)` — REF
- L181: `const embeddingDim = useOnnx ? 384 : (config.embeddingDim ?? 256);` — **CHANGE**

### core/intelligence-engine.js (compiled — mirrors .ts)
- L92: comment — REF
- L94: `const embeddingDim = useOnnx ? 384 : (config.embeddingDim ?? 256);` — **CHANGE**

### core/intelligence-engine.d.ts
- L59: comment — REF

### core/adaptive-embedder.ts
- L27-28: comments `384d`, `384xr`, `rx384` — REF
- L422: `dimension: number = 384` (ProtoNet constructor) — **CHANGE**
- L598: `dimension: number = 384` (ReplayBuffer constructor) — **CHANGE**
- L754: `private dimension: number = 384` (AdaptiveEmbedder) — **CHANGE**
- L998: `baseModel: 'all-MiniLM-L6-v2'` — **CHANGE**

### core/adaptive-embedder.js (compiled — mirrors .ts)
- L28-29: comments — REF
- L311: `dimension = 384` (ProtoNet) — **CHANGE**
- L447: `dimension = 384` (ReplayBuffer) — **CHANGE**
- L583: `this.dimension = 384` (AdaptiveEmbedder) — **CHANGE**
- L770: `baseModel: 'all-MiniLM-L6-v2'` — **CHANGE**

### core/adaptive-embedder.d.ts
- L27-28: comments — REF

### core/neural-embeddings.ts
- L73: `DEFAULT_DIMENSION: 384` — **CHANGE**

### core/neural-embeddings.js (compiled)
- L68: `DEFAULT_DIMENSION: 384` — **CHANGE**

### core/neural-embeddings.d.ts
- L55: `readonly DEFAULT_DIMENSION: 384` — **CHANGE**

### core/neural-perf.ts
- L786: `this.dimension = options.dimension ?? 384` — **CHANGE**

### core/neural-perf.js (compiled)
- L634: `this.dimension = options.dimension ?? 384` — **CHANGE**

### core/router-wrapper.ts
- L58: `dimensions: options.dimensions ?? 384` — **CHANGE**

### core/router-wrapper.js (compiled)
- L46: `dimensions: options.dimensions ?? 384` — **CHANGE**

### services/embedding-service.ts
- L63: `constructor(dimensions: number = 384)` — **CHANGE**

### services/embedding-service.js (compiled)
- L28: `constructor(dimensions = 384)` — **CHANGE**

### workers/native-worker.ts
- L5: comment `384d` — REF
- L127: `dimensions: 384` — **CHANGE**

### workers/native-worker.js (compiled)
- L6: comment — REF
- L105: `dimensions: 384` — **CHANGE**

### workers/native-worker.d.ts
- L5: comment — REF

### workers/benchmark.ts
- L286: `ONNX Model: all-MiniLM-L6-v2` — REF (display string)

### workers/benchmark.js (compiled)
- L208: `ONNX Model: all-MiniLM-L6-v2` — REF (display string)

### bin/cli.js
- L2008: `ONNX Embedding (all-MiniLM-L6-v2)` — REF (display string)

---

## 2. npm/packages/ruvector/src/core/onnx/ (JS Loader Files)

### onnx/loader.js
- L12-13: `'all-MiniLM-L6-v2': { name: 'all-MiniLM-L6-v2'` — **CHANGE** (model registry)
- L14: `dimension: 384` — **CHANGE**
- L18-19: HuggingFace URLs for model & tokenizer — **CHANGE**
- L23: `dimension: 384` (all-mpnet entry) — REF
- L34: `dimension: 384` (bge-small entry) — REF
- L54: `dimension: 384` (gte-small entry) — REF
- L65: `dimension: 384` (nomic entry) — REF
- L77: `export const DEFAULT_MODEL = 'all-MiniLM-L6-v2';` — **CHANGE**
- L277: JSDoc example — REF

### onnx/pkg/loader.js (copy of above)
- Same lines as onnx/loader.js — **CHANGE** (identical file)

---

## 3. Rust Crates

### crates/ruvector-core/src/embeddings.rs
- L29-30: comments — REF
- L154: doc example — REF
- L369, L381, L389: doc comments — REF
- L419, L423: doc examples — REF
- L752: `from_pretrained("sentence-transformers/all-MiniLM-L6-v2")` — TEST
- L781: `from_pretrained("sentence-transformers/all-MiniLM-L6-v2")` — TEST
- L799: same — TEST
- L822: same — TEST

### crates/ruvector-postgres/src/embeddings/mod.rs
- L19, L28-29: comments — REF
- L41: `pub const DEFAULT_MODEL: &str = "all-MiniLM-L6-v2";` — **CHANGE**

### crates/ruvector-postgres/src/embeddings/functions.rs
- L17: doc comment — REF
- L28: `default!(&str, "'all-MiniLM-L6-v2'")` — **CHANGE**
- L64, L76: doc + default — **CHANGE**
- L228: doc example — REF
- L323: doc example — REF
- L350: `ruvector_model_info("all-MiniLM-L6-v2")` — TEST
- L364: `ruvector_embedding_dims("all-MiniLM-L6-v2"), 384` — TEST

### crates/ruvector-postgres/src/embeddings/models.rs
- L8: comment — REF
- L47: `Self::AllMiniLmL6V2 => "all-MiniLM-L6-v2"` — REF (enum variant name mapping)

### crates/ruvector-postgres/sql/embeddings.sql
- L6: `DEFAULT 'all-MiniLM-L6-v2'` — **CHANGE**
- L12: `DEFAULT 'all-MiniLM-L6-v2'` — **CHANGE**

### crates/ruvector-core/src/agenticdb.rs
- L173: doc example — REF

---

## 4. Examples

### examples/onnx-embeddings-wasm/loader.js
- L12-19: Model registry with URLs — **CHANGE** (if this is the canonical loader)
- L77: `export const DEFAULT_MODEL = 'all-MiniLM-L6-v2';` — **CHANGE**
- L277: JSDoc — REF

### examples/onnx-embeddings-wasm/test-full.mjs
- L5, L21, L27, L38: Uses DEFAULT_MODEL import — REF (will follow loader change)

### examples/onnx-embeddings-wasm/parallel-embedder.mjs
- L10, L28: Uses DEFAULT_MODEL import — REF (will follow loader change)

### examples/rust/rag_pipeline.rs
- L22: `options.dimensions = 384;` — **CHANGE**
- L33-74: `mock_embedding(384, ...)` calls — **CHANGE**

### examples/onnx-embeddings/src/lib.rs
- L43, L122, L145: comments and enum mapping — REF

### examples/onnx-embeddings/src/embedder.rs
- L129: comment — REF

### examples/onnx-embeddings/src/main.rs
- L45: comment — REF

### examples/onnx-embeddings/src/model.rs
- L511: test assertion — TEST

### examples/edge/src/p2p/advanced.rs
- L1131: comment — REF

### examples/edge-full/pkg/rvlite/rvlite.js
- L546, L554, L562: comments and config — REF

### examples/edge-full/pkg/generator.html
- L1756: `init(modelName = 'all-MiniLM-L6-v2')` — **CHANGE**

### examples/edge-full/pkg/README.md
- L96, L290: docs — REF

### examples/edge/pkg/generator.html
- L1994: `init(modelName = 'all-MiniLM-L6-v2')` — **CHANGE**

### examples/edge-net/pkg/models/model-optimizer.js
- L46: `id: 'Xenova/all-MiniLM-L6-v2'` — **CHANGE**

### examples/edge-net/pkg/models/models-cli.js
- L78: `huggingface: 'Xenova/all-MiniLM-L6-v2'` — **CHANGE**

### examples/edge-net/pkg/models/registry.json
- L10: `"huggingface": "Xenova/all-MiniLM-L6-v2"` — **CHANGE**

### examples/wasm-vanilla/index.html
- L211: `384` display value — **CHANGE**
- L251: `const DIMENSIONS = 384;` — **CHANGE**

### examples/apify/agentic-synth/src/main.js.backup
- L56: `embeddingDimensions = 384` — **CHANGE**
- L79: `embeddingModel = 'all-MiniLM-L6-v2'` — **CHANGE**
- L367: fallback to model — **CHANGE**

---

## 5. npm/packages/ruvector-extensions/

### src/embeddings.ts
- L649: `Model name or path (default: 'sentence-transformers/all-MiniLM-L6-v2')` — REF
- L682: `model: config.model || 'Xenova/all-MiniLM-L6-v2'` — **CHANGE**
- L693: comment `all-MiniLM-L6-v2 produces 384-dimensional embeddings` — REF

### src/embeddings.js (compiled)
- L438: `model: config.model || 'Xenova/all-MiniLM-L6-v2'` — **CHANGE**
- L447: comment — REF

### src/embeddings.d.ts
- L245: type comment — REF

### src/examples/embeddings-example.ts
- L147: `model: 'Xenova/all-MiniLM-L6-v2'` — **CHANGE**

### src/examples/embeddings-example.js (compiled)
- L125: `model: 'Xenova/all-MiniLM-L6-v2'` — **CHANGE**

---

## 6. Tests

### crates/ruvector-core/tests/embeddings_test.rs
- L220: `from_pretrained("sentence-transformers/all-MiniLM-L6-v2")` — TEST
- L261: same — TEST

---

## 7. Config / Manifest Files

### ui/ruvocal/rvf.manifest.json
- L201: `"embedding_model": "all-MiniLM-L6-v2"` — **CHANGE**

### examples/edge-net/pkg/models/registry.json
- L10: `"huggingface": "Xenova/all-MiniLM-L6-v2"` — **CHANGE**

---

## 8. Other / Helpers

### .claude/helpers/learning-service.mjs
- L65: `model: 'all-MiniLM-L6-v2'` — **CHANGE**
- L469: `modelId: 'all-MiniLM-L6-v2'` — **CHANGE**

---

## 9. Documentation Only (REF — update for accuracy)

- `README.md` L3917, L4820
- `npm/packages/ruvector/README.md` L40, L322, L704, L1302, L1305
- `crates/ruvector-postgres/README.md` L396, L409, L419
- `crates/ruvector-core/docs/EMBEDDINGS.md` L146
- `crates/ruvector-sparse-inference-wasm/README.md` L99
- `crates/ruvector-node/README.md` L225
- `crates/ruvector-node/examples/semantic-search.mjs` L148
- `crates/ruvector-postgres/benches/distance_bench.rs` L9
- `crates/ruvector-postgres/scripts/download_models.rs` L11-12
- `examples/onnx-embeddings-wasm/README.md` L60, L90, L133, L150, L193
- `examples/onnx-embeddings/README.md` L195
- `examples/edge-full/pkg/rvlite/rvlite.d.ts` L59, L63
- `examples/apify/llm/README.md` L841, L889
- `crates/rvf/README.md` L1294
- `docs/guides/AGENTICDB_EMBEDDING_FIX_SUMMARY.md` L145
- `docs/guides/AGENTICDB_EMBEDDINGS_WARNING.md` L104, L111
- `docs/adr/ADR-114-ruvector-core-hash-placeholders.md` L99, L114, L125, L233, L267
- `docs/adr/ADR-074-ruvllm-neural-embeddings.md` L14, L38
- `docs/adr/ADR-115-common-crawl-temporal-compression.md` L282, L340
- `docs/research/models/craftsman-ultra-30b-1bit-ddd.md` L1161
- `ui/ruvocal/docs/adr/ADR-029-HUGGINGFACE-CHAT-UI-CLOUD-RUN.md` L23
- `.claude/skills/agentdb-vector-search/SKILL.md` L28, L304
- `.claude/skills/custom-workers/SKILL.md` L156
- `npm/packages/ruvector-extensions/docs/EMBEDDINGS_SUMMARY.md` L196
- `npm/packages/ruvector-extensions/docs/EMBEDDINGS.md` L126, L322
- `examples/meta-cognition-spiking-neural-network/demos/vector-search/semantic-search.js` L90
- `crates/ruvllm-wasm/src/hnsw_router.rs` L571

---

## Summary

| Category | CHANGE count | REF/comment count | TEST count |
|----------|-------------|-------------------|------------|
| npm/packages/ruvector/src/ (TS source) | ~25 | ~15 | 0 |
| npm/packages/ruvector/src/ (JS compiled) | ~25 | ~15 | 0 |
| npm/packages/ruvector/src/core/onnx/ | ~10 per file (x2 copies) | ~2 | 0 |
| Rust crates | ~5 | ~20 | ~8 |
| Examples | ~15 | ~10 | ~1 |
| Extensions package | ~4 | ~3 | 0 |
| Config/manifest | ~2 | 0 | 0 |
| Helpers | ~2 | 0 | 0 |
| Documentation only | 0 | ~40+ | 0 |

**Primary targets for model swap (functional defaults):**
1. `npm/packages/ruvector/src/core/onnx-embedder.ts` L93 — DEFAULT_MODEL constant
2. `npm/packages/ruvector/src/core/onnx-optimized.ts` L63-69, L185, L189 — model registry + dimension
3. `npm/packages/ruvector/src/core/neural-embeddings.ts` L73 — DEFAULT_DIMENSION
4. `npm/packages/ruvector/src/core/adaptive-embedder.ts` L422, L598, L754, L998 — dimension defaults + baseModel
5. `npm/packages/ruvector/src/core/intelligence-engine.ts` L181 — ONNX dimension branch
6. `npm/packages/ruvector/src/core/router-wrapper.ts` L58 — dimension default
7. `npm/packages/ruvector/src/core/neural-perf.ts` L786 — dimension default
8. `npm/packages/ruvector/src/services/embedding-service.ts` L63 — constructor default
9. `npm/packages/ruvector/src/workers/native-worker.ts` L127 — dimensions config
10. `npm/packages/ruvector/src/core/onnx/loader.js` L12-19, L77 — model registry + DEFAULT_MODEL
11. `crates/ruvector-postgres/src/embeddings/mod.rs` L41 — Rust DEFAULT_MODEL
12. `crates/ruvector-postgres/src/embeddings/functions.rs` L28, L76 — SQL function defaults
13. `crates/ruvector-postgres/sql/embeddings.sql` L6, L12 — SQL DEFAULT values
