# Future tool-calling work

## Current state

- MCP generation uses LEAP-owned prompt templates and contextual JSON Schemas.
- vLLM generates schema-constrained JSON through `StructuredOutputsParams` and the `xgrammar` backend.
- The schema covers LEAP's supported MCP subset: JSON-RPC `tools/call` requests for `select_row`, `select_column`, `group_by`, `sort_by`, and `end`.
- `McpToolCallCodec` strictly validates generated envelopes after decoding.
- The official Python MCP SDK owns transport and tool execution through LEAP's persistent per-worker client.
- LEAP does not currently use vLLM's OpenAI-compatible tool API, automatic tool selection, or model-specific tool parsers.
- MCP prompts are stored in `configs/prompts/iterative_mcp.yaml` and `configs/prompts/cot_mcp.yaml`.

## Findings

- vLLM structured outputs are already the most portable way to enforce valid MCP envelopes across the configured models.
- Native tool calling may produce shorter, more familiar output for models trained for tools, but it does not guarantee better semantic tool selection or arguments.
- Automatic and model-specific tool parsers primarily interpret native model output; they are not a replacement for generation-time schema constraints.
- Using vLLM's OpenAI-compatible API would require changing the current direct-engine architecture and could complicate multi-sample voting, contextual schemas, and two-stage CoT.
- The installed vLLM runtime does not provide a general drop-in MCP client/session manager for LEAP's direct-engine path.
- In a conventional MCP architecture, the model emits an abstract/native tool call and the host constructs the MCP `tools/call` request. LEAP currently asks the model to emit the MCP JSON-RPC envelope directly as an explicit research mode.

## Expected model compatibility

| Model | Native tool-calling outlook | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Good | Candidate for Hermes-style parser and tool-aware chat template |
| `Qwen/Qwen2.5-32B-Instruct` | Good | Recommended first benchmark |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Likely good | Verify Llama 4 Pythonic parser and tokenizer template |
| `openai/gpt-oss-20b` | Good with changes | Native mode must use its Harmony/tool-aware template rather than the current plain-prompt path |
| `openai/gpt-oss-120b` | Good with changes | Same as the 20B model |
| `meta-llama/Llama-2-70b-chat-hf` | Weak or uncertain | Predates current native tool conventions |
| `mistralai/Mistral-7B-Instruct-v0.2` | Weak or uncertain | Parser availability does not guarantee matching model training/template |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Weak or uncertain | Same concern |
| Non-instruct Llama/Mixtral models | Poor | No reliable native tool behavior expected |
| `gpt2` | Unsupported in practice | Retain only for structural tests |

Native tool support requires all three of the following to agree:

1. The model was trained for tool calling.
2. Its tokenizer chat template accepts and renders tool definitions.
3. The configured vLLM parser matches the model's emitted syntax.

## Proposed implementation

- [ ] Keep the existing `output_format: mcp` behavior unchanged as the portable, schema-constrained research path.
- [ ] Add a separate experimental `output_format: native_tool` rather than replacing MCP mode.
- [ ] Add explicit per-model tool capabilities to `configs/models.yaml`, including whether native tools are supported, the chat-template strategy, and the vLLM parser name.
- [ ] Reject `native_tool` during configuration loading when the selected model has no declared compatible parser/template.
- [ ] Start with `Qwen/Qwen2.5-32B-Instruct`; verify its exact tokenizer template and parser against the pinned vLLM version before implementation.
- [ ] Render native tool definitions through the model's tool-aware chat template.
- [ ] Normalize parsed native calls into the existing `McpToolCall` representation.
- [ ] Reuse the persistent `McpTableClient`, stateless table payload, retry policy, and MCP server without modification.
- [ ] Preserve enabled-action filtering, contextual row/column restrictions, sampling, voting, and CoT behavior.
- [ ] Decide explicitly whether native mode should use vLLM's OpenAI API server or remain on the direct engine; do not introduce HTTP solely for parsing convenience.
- [ ] If using the OpenAI API, prototype support for `n` candidates, dynamic tools, required tool choice, cancellation, profiling, and two-stage CoT before adopting it.
- [ ] Keep model-specific native output out of persisted protocol data by normalizing it at the generation boundary.
- [ ] Document that native tool calls are host-translated into MCP requests and are not themselves MCP wire messages.

## Required tests

- [ ] Verify each enabled model's chat template accepts tool definitions.
- [ ] Verify the configured parser can parse representative single and parallel tool-call outputs.
- [ ] Verify unsupported models fail configuration validation clearly.
- [ ] Verify native calls normalize to the same `Action` and `McpToolCall` values as JSON and MCP-envelope modes.
- [ ] Verify table data is still injected only by LEAP and never generated as a model argument.
- [ ] Verify CoT selection is not dispatched and only the complete argument-stage call reaches MCP.
- [ ] Verify malformed or textual native output follows the existing invalid-candidate and fallback behavior.
- [ ] Verify function, JSON, and MCP-envelope modes are unchanged.

## Accuracy and performance evaluation

Benchmark at least these modes with identical models, tables, actions, sampling counts, temperatures, and seeds where supported:

1. MCP envelope with JSON Schema constraints.
2. Flat JSON with JSON Schema constraints.
3. Native tool calling with available constraints.
4. Native tool calling without generation constraints.

Record:

- End-to-end answer accuracy.
- Tool-selection accuracy.
- Argument accuracy.
- Candidate validity before post-generation filtering.
- Fallback frequency.
- Generated token count.
- Generation latency.
- MCP startup, warm-call, and total transformation timings separately.

Do not assume an accuracy gain. The expected advantages of native tools are lower output-token overhead, closer alignment with tool-trained models, and potentially better unconstrained validity. Adoption should require measured improvement without regressions to lifecycle behavior or experiment reproducibility.
