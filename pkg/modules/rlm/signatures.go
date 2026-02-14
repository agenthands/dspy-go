package rlm

import (
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// =============================================================================
// DSPy-Native Signature Definitions
// =============================================================================

// RLMSignature creates the main RLM module signature.
// This is the outer interface: takes context + query, returns answer.
func RLMSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context",
				core.WithDescription("The context data to analyze (can be very large)"),
			)},
			{Field: core.NewField("query",
				core.WithDescription("The question to answer about the context"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("answer",
				core.WithDescription("The final answer to the query"),
			)},
		},
	).WithInstruction("Analyze the context using iterative code exploration to answer the query.")
}

// IterationSignature defines the signature for each RLM iteration.
// This powers the inner loop where the LLM decides what to do next.
func IterationSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context_info",
				core.WithDescription("Summary of the context (type, size, preview)"),
			)},
			{Field: core.NewField("query",
				core.WithDescription("The original question to answer"),
			)},
			{Field: core.NewField("history",
				core.WithDescription("Previous code executions and their results"),
			)},
			{Field: core.NewField("repl_state",
				core.WithDescription("Current REPL variable state"),
			)},
			{Field: core.NewField("iteration",
				core.WithDescription("Current iteration progress (X of Y)"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("reasoning",
				core.WithDescription("Step-by-step thinking about what to do next"),
				core.WithCustomPrefix("Reasoning:"),
			)},
			{Field: core.NewField("action",
				core.WithDescription("The action type: 'explore', 'query', 'compute', 'subrlm', or 'final'"),
				core.WithCustomPrefix("Action:"),
			)},
			{Field: core.NewField("code",
				core.WithDescription("Go code to execute (if action is explore/query/compute)"),
				core.WithCustomPrefix("Code:"),
			)},
			{Field: core.NewField("subquery",
				core.WithDescription("The query for the sub-RLM (if action is 'subrlm')"),
				core.WithCustomPrefix("SubQuery:"),
			)},
			{Field: core.NewField("answer",
				core.WithDescription("The final answer (if action is 'final')"),
				core.WithCustomPrefix("Answer:"),
			)},
		},
	).WithInstruction(`You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query. You should check the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A 'llm_query' function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment. Use this for recursive calls!
3. A 'llm_query_batched' function that allows you to query multiple prompts concurrently: llm_query_batched(prompts []string) []string. This is much faster than sequential calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
4. A 'SHOW_VARS()' function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
5. Standard Go functions and the ability to use fmt.Println() to view output.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Go code in the REPL environment, write to the 'Code' field. For example, say we want our recursive model to search for the magic number in the context, and the context is very long, so we want to chunk it:
Reasoning: The context is very long, so I want to chunk it.
Action: query
Code:
` + "```go" + `
chunk := context[:10000]
answer := llm_query("What is the magic number in the context? Here is the chunk: " + chunk)
fmt.Println(answer)
` + "```" + `

As another example, when the context isn't that long, a simple but viable strategy is to combine chunks and recursively query an LLM over them using llm_query_batched for concurrent processing:
` + "```go" + `
chunkSize := len(context) / 10
var prompts []string
for i := 0; i < 10; i++ {
    start := i * chunkSize
    end := (i + 1) * chunkSize
    if i == 9 { end = len(context) }
    prompts = append(prompts, fmt.Sprintf("Try to answer: %s. Here is the context chunk:\n%s", query, context[start:end]))
}
answers := llm_query_batched(prompts)
for i, a := range answers { fmt.Printf("Chunk %d: %s\n", i, a) }
` + "```" + `

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function. You have two options:
1. Use FINAL(your_answer_variable) to provide the answer directly in a Go code block
2. Use FINAL_VAR("variable_name") to return a variable you have created in the REPL environment as your final output

WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a code block FIRST, then call FINAL_VAR in a SEPARATE step.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.

CODE RULES:
- DO NOT use 'import' statements.
- DO NOT use 'type' or top-level 'func' declarations.
- Every variable must be declared with := before use.

OUTPUT FORMAT (REQUIRED):
Each response MUST contain: Reasoning, Action, Code, and Answer.`)
}

// SubQuerySignature defines the signature for sub-LLM queries.
// This is used by Query() and QueryBatched() internally.
func SubQuerySignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("prompt",
				core.WithDescription("The analysis prompt"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("response",
				core.WithDescription("The analysis result"),
			)},
		},
	).WithInstruction("Analyze the given content and provide a concise, accurate response.")
}

// ChunkAnalysisSignature for analyzing individual chunks of large contexts.
func ChunkAnalysisSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("chunk",
				core.WithDescription("A portion of the larger context"),
			)},
			{Field: core.NewField("query",
				core.WithDescription("What to look for in this chunk"),
			)},
			{Field: core.NewField("chunk_index",
				core.WithDescription("Which chunk this is (e.g., '3 of 10')"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("findings",
				core.WithDescription("Relevant findings from this chunk"),
			)},
			{Field: core.NewField("confidence",
				core.WithDescription("Confidence level: high, medium, low"),
			)},
		},
	).WithInstruction("Analyze this chunk for information relevant to the query. Be concise but thorough.")
}

// SynthesisSignature for combining results from multiple chunk analyses.
func SynthesisSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("query",
				core.WithDescription("The original question"),
			)},
			{Field: core.NewField("chunk_results",
				core.WithDescription("Results from analyzing each chunk"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("synthesis",
				core.WithDescription("Combined analysis of all chunks"),
			)},
			{Field: core.NewField("answer",
				core.WithDescription("The final answer based on all evidence"),
			)},
		},
	).WithInstruction("Synthesize the chunk analysis results into a coherent answer.")
}

// =============================================================================
// Example Demos for Few-Shot Learning
// =============================================================================

// IterationDemos provides few-shot examples for the iteration module.
func IterationDemos() []core.Example {
	return []core.Example{
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 150000 chars",
				"query":        "What is the secret code?",
				"history":      "",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "This is a large context (150K chars). I should first explore its structure before diving into analysis.",
				"action":    "explore",
				"code":      "fmt.Println(\"Length:\", len(context))\nfmt.Println(\"Preview:\", context[:500])",
				"subquery":  "",
				"answer":    "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 150000 chars",
				"query":        "What is the secret code?",
				"history":      "Explored: Length=150000, Preview shows JSON array of messages",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Context is too large to process at once. I'll chunk it and use QueryBatched for parallel analysis.",
				"action":    "query",
				"code": `chunkSize := len(context) / 5
var prompts []string
for i := 0; i < 5; i++ {
    start, end := i*chunkSize, (i+1)*chunkSize
    if i == 4 { end = len(context) }
    prompts = append(prompts, fmt.Sprintf("Find any secret codes in this text. Return ONLY the code if found, or 'none' if not found: %s", context[start:end]))
}
results := QueryBatched(prompts)
for i, r := range results { fmt.Printf("Chunk %d: %s\n", i, r) }`,
				"subquery": "",
				"answer":   "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 150000 chars",
				"query":        "What is the secret code?",
				"history":      "Chunk 0: none\nChunk 1: ALPHA-7892\nChunk 2: none\nChunk 3: none\nChunk 4: none",
				"repl_state":   "context: <loaded>, results: [none, ALPHA-7892, none, none, none]",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Found the secret code in chunk 1: ALPHA-7892. This is the answer.",
				"action":    "final",
				"code":      "",
				"subquery":  "",
				"answer":    "ALPHA-7892",
			},
		},
		// Example demonstrating FINAL() called immediately after Query()
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 5000 chars",
				"query":        "What is the label: correct or incorrect?",
				"history":      "Explored: Length=5000, Preview shows labeled data",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Small context that fits in one Query call. I'll ask the sub-LLM to determine the label and IMMEDIATELY call FINAL with the result.",
				"action":    "query",
				"code": `answer := Query("What is the label in this text? Return ONLY 'correct' or 'incorrect': " + context)
FINAL(answer)`,
				"subquery": "",
				"answer":   "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 800000 chars",
				"query":        "Count all error messages in the logs",
				"history":      "",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Context is 800K chars. Sub-LLMs handle ~500K, so I'll use regex to filter first, then batch ~200K per query for efficiency.",
				"action":    "explore",
				"code": `errorRe := regexp.MustCompile("(?i)error|exception|failed")
matches := errorRe.FindAllString(context, -1)
fmt.Println("Total potential errors:", len(matches))
fmt.Println("Sample:", context[:1000])`,
				"subquery": "",
				"answer":   "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 1200000 chars",
				"query":        "Summarize the main themes in this document collection",
				"history":      "Explored: 1.2M chars, appears to be multiple documents separated by ---",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "At 1.2M chars, I MUST use QueryWith or QueryRaw - Query() would overflow! I'll split by document separator and use QueryWith for each batch.",
				"action":    "query",
				"code": `docs := strings.Split(context, "---")
fmt.Println("Found", len(docs), "documents")

var results []string
var batch string
for _, doc := range docs {
    if len(batch)+len(doc) > 150000 && batch != "" {
        // Use QueryWith to send ONLY this batch, not full context
        r := QueryWith(batch, "Identify main themes in these documents")
        results = append(results, r)
        batch = ""
    }
    batch += doc + "\n---\n"
}
if batch != "" {
    r := QueryWith(batch, "Identify main themes in these documents")
    results = append(results, r)
}
for i, r := range results { fmt.Printf("Batch %d: %s\n", i, r) }`,
				"subquery": "",
				"answer":   "",
			},
		},
		// Example using FindRelevant for semantic search on large context
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 800000 chars",
				"query":        "Find all authentication-related code",
				"history":      "Explored: 800K chars of Go source code, too large for single Query",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "At 800K chars, I cannot use Query() as it would overflow. I'll use FindRelevant() to get semantically relevant chunks, then QueryWith() to analyze each.",
				"action":    "query",
				"code": `// Find chunks most relevant to authentication
chunks := FindRelevant("authentication login password session token", 10)
fmt.Println("Found", len(chunks), "relevant chunks")

var authCode []string
for i, chunk := range chunks {
    // Use QueryWith to analyze just this chunk
    result := QueryWith(chunk, "Extract any authentication-related code (login, password, session, token handling). Return the code snippets or 'none' if not found.")
    if result != "none" && result != "" {
        authCode = append(authCode, fmt.Sprintf("Chunk %d:\n%s", i, result))
    }
}
fmt.Println("Auth code found in", len(authCode), "chunks")
for _, code := range authCode { fmt.Println(code) }`,
				"subquery": "",
				"answer":   "",
			},
		},
		// Example showing synthesis with QueryRaw
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 800000 chars",
				"query":        "Find all authentication-related code",
				"history":      "Found auth code in 4 chunks: JWT token validation, password hashing, session middleware, OAuth handler",
				"repl_state":   "context: <loaded>, authCode: [4 code snippets]",
			},
			Outputs: map[string]interface{}{
				"reasoning": "I have 4 authentication code snippets. Now I'll synthesize them using QueryRaw (no context needed) and immediately FINAL the answer.",
				"action":    "query",
				"code": `summary := strings.Join(authCode, "\n\n---\n\n")
// Use QueryRaw since we're passing our own content, not the original context
answer := QueryRaw("Summarize these authentication code findings into a coherent explanation:\n\n" + summary)
FINAL(answer)`,
				"subquery": "",
				"answer":   "",
			},
		},
		// Example using Search and GetDocument tools for information retrieval
		{
			Inputs: map[string]interface{}{
				"context_info": "empty context, access to Search/GetDocument tools",
				"query":        "What is the capital of France?",
				"history":      "",
				"repl_state":   "context: <empty>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "I need to find information about France's capital. I'll use the Search tool.",
				"action":    "query",
				"code": `results := Search("capital of France")
fmt.Println(results)`,
				"subquery": "",
				"answer":   "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "empty context, access to Search/GetDocument tools",
				"query":        "What is the capital of France?",
				"history":      "Search results: [1] DocID: doc_123\nParis is the capital of France...",
				"repl_state":   "context: <empty>, results: ...",
			},
			Outputs: map[string]interface{}{
				"reasoning": "The search result mentions doc_123. I'll retrieve the full document to be sure.",
				"action":    "query",
				"code": `doc := GetDocument("doc_123")
fmt.Println(doc)`,
				"subquery": "",
				"answer":   "",
			},
		},
		// Example using subrlm action for complex multi-step analysis
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 800000 chars",
				"query":        "Analyze the authentication system and identify security issues",
				"history":      "Explored: 800K chars of source code, complex authentication system with multiple components",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "This is a complex analysis requiring multiple steps. I'll spawn a sub-RLM to first find all authentication-related code, then analyze it for vulnerabilities in a follow-up.",
				"action":    "subrlm",
				"code":      "",
				"subquery":  "Find all authentication-related code sections in this codebase including login, session handling, token validation, and password management",
				"answer":    "",
			},
		},
		// Example demonstrating CodeQA-like logic for complex reasoning
		{
			Inputs: map[string]interface{}{
				"context_info": "Python function (html_to_xhtml)",
				"query":        "How do all tags in an html tree convert to xhtml ?",
				"history":      "",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "I need to analyze how the tags are processed in the html_to_xhtml function. I'll use llm_query to understand the logic specifically for tag conversion.",
				"action":    "query",
				"code": `answer := llm_query("Identify the logic responsible for converting tags to XHTML in this code: " + context)
fmt.Println(answer)
FINAL(answer)`,
				"subquery": "",
				"answer":   "",
			},
		},
	}
}

// SubQueryDemos provides few-shot examples for sub-LLM queries.
func SubQueryDemos() []core.Example {
	return []core.Example{
		{
			Inputs: map[string]interface{}{
				"prompt": "Find any email addresses in this text: Contact us at support@example.com or sales@example.com",
			},
			Outputs: map[string]interface{}{
				"response": "support@example.com, sales@example.com",
			},
		},
		{
			Inputs: map[string]interface{}{
				"prompt": "What is the main topic of this text? Return one word: The quick brown fox jumps over the lazy dog.",
			},
			Outputs: map[string]interface{}{
				"response": "animals",
			},
		},
	}
}
