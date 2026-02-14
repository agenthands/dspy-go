package rlm

import (
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// MangleIterationSignature defines the signature for RLM iterations using Mangle Datalog.
func MangleIterationSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context_info",
				core.WithDescription("Summary of the loaded facts (predicates, counts)"),
			)},
			{Field: core.NewField("query",
				core.WithDescription("The original question to answer"),
			)},
			{Field: core.NewField("history",
				core.WithDescription("Previous Datalog queries and their results"),
			)},
			{Field: core.NewField("repl_state",
				core.WithDescription("Current environment state (loaded facts)"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("reasoning",
				core.WithDescription("Step-by-step thinking about what Datalog query to run"),
				core.WithCustomPrefix("Reasoning:"),
			)},
			{Field: core.NewField("action",
				core.WithDescription("The action type: 'query', 'final'"),
				core.WithCustomPrefix("Action:"),
			)},
			{Field: core.NewField("code",
				core.WithDescription("Mangle Datalog code to execute (rules + query)"),
				core.WithCustomPrefix("Code:"),
			)},
			{Field: core.NewField("answer",
				core.WithDescription("The final answer (if action is 'final')"),
				core.WithCustomPrefix("Answer:"),
			)},
		},
	).WithInstruction(`You are reasoning about a code snippet using a Mangle Datalog engine.
The code has been parsed into an Abstract Syntax Tree (AST) represented as Datalog facts.
Your goal is to answer the user's query by analyzing the code's logic, structure, and data flow.
Use Datalog rules to traverse the AST and find relevant code patterns (e.g., function definitions, loops, variable assignments) that implement the logic described in the query.

OUTPUT FORMAT (REQUIRED):
Reasoning: <your thinking>
Action: <one of: query, final>
Code: <Mangle Datalog code>
Answer: <final answer if action is final>

MANGLE DATALOG SYNTAX:
- Facts: pred(arg1, arg2).
- Rules: head(X) :- body(X, Y), other(Y).
- Query: Declared via 'Query' predicate or implicit last rule? 
  Result of execution is the content of the 'result' relation if defined, or the output of the query.
  Ideally, define a 'result(X)' rule that captures the answer.


MANGLE AST SCHEMA (when parsing code):
- root(RootID).
- node(ID, Type, StartByte, EndByte).
- child(ParentID, Index, ChildID).
- field(ParentID, FieldName, ChildID).
- text(ID, Content).

SEMANTIC PREDICATES (available via enrichment rules):
- name_of(NodeID, Name): Extracts identifier name.
- assignment(Target, ValueNode): Matches 'target = value'.
- call(CallID, FunctionName): Matches 'func()'.
- attribute_access(Object, Attribute): Matches 'obj.attr'.
- flows_to(Source, Sink): Tracks data movement via assignments.
- concat(Left, Right, Result): Matches 'A + B' string concatenation (often used for prefixing).
- transformation(Input, Output): Identifies when 'Output' is derived from 'Input'.


EXAMPLE:
Reasoning: I need to find the function name.
Action: query
Code:
` + "```prolog" + `
match(Name) :- root(R), node(R, "module", _, _), child(R, _, F), node(F, "function_definition", _, _), field(F, "name", N), text(N, Name).
result(X) :- match(X).
` + "```" + `
Answer:

available predicates are listed in context_info.
`)
}

// MangleIterationDemos provides few-shot examples for Mangle iterations.
func MangleIterationDemos() []core.Example {
	return []core.Example{
		{
			Inputs: map[string]interface{}{
				"context_info": "Facts: user(id, name), transaction(id, amount)",
				"query":        "Who spent more than 100?",
				"history":      "",
				"repl_state":   "Loaded 50 facts.",
			},
			Outputs: map[string]interface{}{
				"reasoning": "I need to join user and transaction logic.",
				"action":    "query",
				"code": `high_spender(Name) :- user(Id, Name), transaction(TId, Amount), Amount > 100.
result(N) :- high_spender(N).`,
				"answer": "",
			},
		},
	}
}
