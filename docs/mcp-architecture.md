# MCP Table-Transformation Architecture

This document compares LEAP's original function/JSON operation path with the MCP-backed path added as `generation.output_format: mcp`.

## Before: local operation parsing and execution

In the original architecture, the model emitted either function syntax or a named JSON object. LEAP parsed the output into an internal `Action` and applied that action directly through the in-process action registry.

```mermaid
flowchart LR
    C[GenerationConfig<br/>output_format: function or json]
    P[PromptBuilder]
    L[vLLM model]
    D{Output format}
    FP[Function parser / grammar]
    JP[JSON codec / schema]
    A[Internal Action]
    R[In-process ActionRegistry]
    T[(Current Table)]
    NT[(New Table)]
    H[Action history]
    E[Answer extractors]

    C --> P
    P --> L
    L --> D
    D -->|function| FP
    D -->|json| JP
    FP --> A
    JP --> A
    A --> R
    T --> R
    R --> NT
    A --> H
    NT -->|next step| P
    NT -->|after end| E
    H --> P
```

### Before: information flow

```mermaid
sequenceDiagram
    participant Strategy as Generation strategy
    participant Prompt as PromptBuilder
    participant Model as vLLM model
    participant Parser as Function/JSON parser
    participant Registry as ActionRegistry

    Strategy->>Prompt: question + current table + history
    Prompt->>Model: operation prompt
    Model-->>Parser: f_action(...) or JSON object
    Parser-->>Strategy: Action(name, arguments)
    Strategy->>Registry: apply(action, current table)
    Registry-->>Strategy: new Table
    Strategy->>Strategy: append history and continue
```

The table never crossed a process or protocol boundary. `Action.apply_to_table()` delegated directly to the registered Python action implementation.

## After: selectable MCP execution path

Function and JSON modes remain available and retain their original execution path. MCP is an additional output mode. In MCP mode, the model emits an official JSON-RPC `tools/call` envelope, LEAP validates and normalizes it, and the selected transformation is executed by a stateless MCP server over stdio.

```mermaid
flowchart LR
    C[GenerationConfig<br/>output_format: function, json, or mcp]
    P[PromptBuilder]
    L[vLLM model]
    D{Output format}

    subgraph Existing[Existing paths, unchanged]
        FP[Function parser / grammar]
        JP[JSON codec / schema]
        LR[In-process ActionRegistry]
    end

    subgraph MCPPath[MCP path]
        MP[McpToolCallCodec<br/>validate JSON-RPC envelope]
        MS[MCP contextual schema]
        V[Candidate validation / voting]
        MA[Shared generation loop]
        MC[Per-worker McpTableClient<br/>serialized broker]
        STDIO[Persistent MCP stdio session]
        SV[Stateless FastMCP server]
        TOOL[Per-operation MCP tool]
    end

    T[(Current Table)]
    NT[(New Table)]
    E[Answer extractors]

    C --> P
    P --> L
    L --> D

    D -->|function| FP --> LR
    D -->|json| JP --> LR
    T --> LR --> NT

    D -->|mcp| MP
    MS -. constrained decoding .-> L
    MP --> V --> MA
    MA --> MC
    T -->|inject complete table state| MC
    MC -->|lazy initialize once| STDIO
    STDIO --> SV --> TOOL
    TOOL -->|complete new table state| SV
    SV --> STDIO --> MC --> NT

    NT -->|next transformation| P
    NT -->|after end| E
```

## MCP request and response boundary

The model-facing request contains the operation name and its arguments, but not the table. LEAP owns the current table and injects it before invoking the server.

```mermaid
flowchart TB
    M[Model output]
    Q[Strict tools/call envelope<br/>jsonrpc + id + method + params]
    N[LEAP normalization<br/>validate tool and arguments]
    I[Inject current table]
    B[Per-worker serialized broker]
    W[Persistent MCP client call]
    S[Stateless MCP tool]
    O[Structured result<br/>action + table + terminated]
    U[LEAP updates current table]

    M --> Q --> N --> I --> B --> W --> S --> O --> U
```

Example model output:

```json
{
  "jsonrpc": "2.0",
  "id": "step",
  "method": "tools/call",
  "params": {
    "name": "sort_by",
    "arguments": {
      "column": "Score",
      "order": "desc"
    }
  }
}
```

LEAP invokes the MCP tool with the runtime-owned table added to its arguments:

```json
{
  "name": "sort_by",
  "arguments": {
    "table": {
      "columns": ["Name", "Score"],
      "rows": [["Ada", "2"], ["Grace", "1"]]
    },
    "column": "Score",
    "order": "desc"
  }
}
```

The server returns the authoritative table state:

```json
{
  "action": "sort_by('Score', 'desc')",
  "table": {
    "columns": ["Name", "Score"],
    "rows": [["Ada", "2"], ["Grace", "1"]]
  },
  "terminated": false
}
```

## Persistent worker lifecycle

Each vLLM worker enters one `McpTableClient` context before its request loop. Entering the context creates only a lightweight broker task. The broker lazily launches and initializes the stdio server on the first MCP call, serializes calls from concurrent inference tasks, and retains the session until worker shutdown. Function/JSON-only workers never launch the subprocess.

```mermaid
sequenceDiagram
    participant Worker as vLLM worker
    participant Broker as MCP broker task
    participant Server as FastMCP subprocess

    Worker->>Broker: enter client context (no subprocess)
    Note over Worker,Broker: Function/JSON calls do not initialize MCP
    Worker->>Broker: first MCP operation
    Broker->>Server: launch + initialize once
    Server-->>Broker: initialized session
    loop All later MCP operations
        Worker->>Broker: enqueue operation
        Broker->>Server: call_tool on existing session
        Server-->>Broker: table state
        Broker-->>Worker: result + timings
    end
    Worker->>Broker: worker shutdown
    Broker->>Server: close stdin/session
    Broker-->>Worker: cleanup complete
```

If transport startup or a warm call fails, the broker closes the session, reconnects, and retries that stateless call once. Tool validation errors are returned without reconnecting or retrying.

## Iterative MCP flow

Iterative mode generates one complete MCP request per transformation step.

```mermaid
sequenceDiagram
    participant Strategy as Iterative strategy
    participant Model as vLLM model
    participant Codec as McpToolCallCodec
    participant Client as Persistent MCP broker
    participant Server as FastMCP server

    Strategy->>Model: table + question + history + tool shapes
    Model-->>Codec: complete tools/call request
    Codec-->>Strategy: validated action and arguments
    Strategy->>Client: winning operation + current table
    Client->>Server: call_tool on persistent session
    Server-->>Client: structured new table state
    Client-->>Strategy: action + Table + termination status
    Strategy->>Strategy: record action and continue or terminate
```

## Two-stage CoT MCP flow

CoT retains its two-stage generation behavior. The selection-stage envelope is used for planning and is not dispatched because its `arguments` object is intentionally empty. The complete second-stage envelope is validated and executed through MCP.

```mermaid
sequenceDiagram
    participant Strategy as CoT strategy
    participant Model as vLLM model
    participant Codec as McpToolCallCodec
    participant Client as Persistent MCP broker
    participant Server as FastMCP server

    Strategy->>Model: Phase 1: select the next tool
    Model-->>Codec: tools/call shape with selected name and arguments: {}
    Codec-->>Strategy: selected tool name
    Strategy->>Model: Phase 2: emit complete request for selected tool
    Model-->>Codec: tools/call request with complete arguments
    Codec-->>Strategy: validated action
    Strategy->>Client: selected operation + current table
    Client->>Server: call_tool on persistent session
    Server-->>Client: structured new table state
    Client-->>Strategy: action + Table + termination status
```

## Tool surface

| MCP tool | Input besides `table` | Result |
|---|---|---|
| `select_row` | `rows: list[str]` | Selected rows |
| `select_column` | `columns: list[str]` | Selected columns |
| `add_column` | `column: str`, `values: list[str]` | Table with one new value per row |
| `group_by` | `column: str` | Group counts |
| `sort_by` | `column: str`, `order: "asc" \| "desc"` | Sorted rows |
| `end` | None | Unchanged table with `terminated: true` |

`add_column` validates that the new column name is unused and that the values list contains exactly one string per table row.

## State ownership and compatibility

- **LEAP runtime owns table state:** the persistent server process does not retain tables or operation state between calls.
- **The MCP server owns transformation execution in MCP mode:** its returned table replaces the runtime's current table.
- **The model never owns or transmits hidden server state:** it chooses only the tool and operation arguments.
- **Function and JSON compatibility is preserved:** those modes continue to parse into `Action` and execute locally.
- **Termination remains compatible:** MCP `end` is executed by the server, after which LEAP runs the existing answer extractors.

## Relevant implementation files

| Responsibility | File |
|---|---|
| MCP envelope codec and schemas | `leap/mcp/protocol.py` |
| MCP stdio client adapter | `leap/mcp/client.py` |
| Stateless FastMCP server and tools | `leap/mcp/server.py` |
| Model prompts | `leap/generation/prompt_builder.py` |
| Candidate generation and parsing | `leap/generation/sampling.py` |
| Table-state integration | `leap/generation/strategies.py` |
| Configuration validation | `leap/config/loader.py` |
| Protocol and schema tests | `tests/test_mcp_protocol.py` |
| Client lifecycle and recovery tests | `tests/test_mcp_client.py` |
| MCP server round-trip tests | `tests/test_mcp_server.py` |
| Worker lifecycle tests | `tests/test_vllm_server.py` |
