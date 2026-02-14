package rlm

// GetPythonEnrichmentRules returns a set of Datalog rules
// to extract higher-level semantics from Tree-sitter Python AST facts.
func GetPythonEnrichmentRules() string {
	return `
# --- Semantic Enrichment Rules for Python ---

# Helper: name_of(NodeID, Name)
# Extracts the text content of an identifier node.
name_of(N, Name) :- node(N, "identifier", _, _), text(N, Name).

# Rule: assignment(Target, Value)
# Matches 'target = value' assignments.
assignment(TargetName, ValueID) :- 
    node(A, "assignment", _, _),
    child(A, 0, T), name_of(T, TargetName),
    child(A, 2, ValueID).  # Python assignment structure: target = value (index 0 and 2 usually)
    # Note: Index might vary based on tree-sitter grammar version, need to verify.
    # Actually, field "left" and "right" are better.

assignment(TargetName, ValueID) :-
    node(A, "assignment", _, _),
    field(A, "left", T), name_of(T, TargetName),
    field(A, "right", ValueID).

# Rule: function_call(Caller, FunctionName)
# Matches 'func(args)' calls.
call(CallID, FunctionName) :-
    node(CallID, "call", _, _),
    field(CallID, "function", FuncNode),
    name_of(FuncNode, FunctionName).
    
# Rule: method_call(CallID, Object, MethodName)
# Matches 'obj.method(args)'
call(CallID, MethodName) :-
    node(CallID, "call", _, _),
    field(CallID, "function", AttrNode),
    node(AttrNode, "attribute", _, _),
    field(AttrNode, "attribute", MethodNode),
    name_of(MethodNode, MethodName).

method_call(CallID, ObjectName, MethodName) :-
    node(CallID, "call", _, _),
    field(CallID, "function", AttrNode),
    node(AttrNode, "attribute", _, _),
    field(AttrNode, "object", ObjNode),
    name_of(ObjNode, ObjectName),
    field(AttrNode, "attribute", MethodNode),
    name_of(MethodNode, MethodName).

# Rule: for_loop(Target, Iterator)
# Matches 'for target in iterator:'
for_loop(TargetName, IteratorID) :-
    node(F, "for_statement", _, _),
    field(F, "left", TargetNode), name_of(TargetNode, TargetName),
    field(F, "right", IteratorID).

# Rule: attribute_access(Object, Attribute)
# Matches 'obj.attr'
attribute_access(ObjectName, AttributeName) :-
    node(A, "attribute", _, _),
    field(A, "object", ObjNode), name_of(ObjNode, ObjectName),
    field(A, "attribute", AttrNode), name_of(AttrNode, AttributeName).

# Rule: string_literal(ID, Value)
string_literal(S, Value) :-
    node(S, "string", _, _),
    text(S, RawValue),
    # Rough approximation to strip quotes, Datalog string manipulation is limited
    # We'll just provide the raw text for now.
    text(S, Value). # Assuming text/2 extracts content

# --- Data Flow Analysis (Naive) ---

# Rule: flows_to(Source, Sink)
# Tracks data movement via assignments.
# If A = B, data flows from B to A.
flows_to(ValID, TargetID) :-
    assignment(TargetName, ValID),
    name_of(TargetNode, TargetName),
    node(TargetNode, "identifier", _, _),
    # Ideally link TargetName to specific usage, but for now link by name
    text(TargetID, TargetName). # Mangle specific: if ID has name

# Rule: concat(Left, Right, Result)
# Matches 'prefix + tag' patterns (binary_operator with +)
concat(LeftID, RightID, OpID) :-
    node(OpID, "binary_operator", _, _),
    field(OpID, "left", LeftID),
    field(OpID, "right", RightID),
    child(OpID, 1, OpSymbol), text(OpSymbol, "+").

# Rule: transformation(Input, Output)
# Captures 'output = ... input ...'
transformation(InputID, OutputID) :-
    assignment(OutputName, ExprID),
    # Check if ExprID involves InputID
    contains_usage(ExprID, InputID).

contains_usage(Parent, Child) :- child(Parent, _, Child).
contains_usage(Parent, Descendant) :- child(Parent, _, Child), contains_usage(Child, Descendant).


`
}
