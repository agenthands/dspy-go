package rlm

import (
	"context"
	"fmt"
	"strings"
	"unsafe"

	"github.com/google/mangle/ast"
	"github.com/google/mangle/parse"
	sitter "github.com/tree-sitter/go-tree-sitter"
	treeSitterGo "github.com/tree-sitter/tree-sitter-go/bindings/go"
	treeSitterJava "github.com/tree-sitter/tree-sitter-java/bindings/go"
	treeSitterPython "github.com/tree-sitter/tree-sitter-python/bindings/go"
)

// ParseCode parses source code and returns Mangle Datalog facts representing the AST.
func ParseCode(ctx context.Context, code string, language string) ([]ast.Clause, string, error) {
	var langPtr unsafe.Pointer
	language = strings.ToLower(language)

	switch language {
	case "python":
		langPtr = treeSitterPython.Language()
	case "go":
		langPtr = treeSitterGo.Language()
	case "java":
		langPtr = treeSitterJava.Language()
	default:
		return nil, "", fmt.Errorf("unsupported language: %s", language)
	}

	lang := sitter.NewLanguage(langPtr)
	parser := sitter.NewParser()
	parser.SetLanguage(lang)
	defer parser.Close()

	sourceBytes := []byte(code)
	tree := parser.Parse(sourceBytes, nil)
	defer tree.Close()

	rootNode := tree.RootNode()
	if rootNode.HasError() {
		// We might still want partial results, but for now returning error is safer
		// Actually, HasError returns true if ANY node in the tree is an error/missing.
		// For robustness, we should warn but continue.
	}

	walker := &ASTWalker{
		source: sourceBytes,
		facts:  make([]ast.Clause, 0),
		nextID: 1,
	}

	rootID := walker.generateID()
	cursor := tree.Walk()
	defer cursor.Close()

	walker.walk(cursor, rootID)

	// Add root fact: root(id)
	rootAtom := ast.NewAtom("root", ast.Constant(ast.String(rootID)))
	walker.facts = append(walker.facts, ast.NewClause(rootAtom, nil))

	info := fmt.Sprintf("Parsed %s code. Loaded %d AST facts.", language, len(walker.facts))

	// Inject semantic enrichment rules
	if language == "python" {
		ruleStr := GetPythonEnrichmentRules()
		// We actually need to return these rules alongside facts or append them.
		// Mangle facts are just clauses. Rules are also clauses (with body).
		// We need to parse the rule string into []ast.Clause
		ruleSource, err := parse.Unit(strings.NewReader(ruleStr))
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse enrichment rules: %w", err)
		}
		walker.facts = append(walker.facts, ruleSource.Clauses...)
		info += fmt.Sprintf(" Added %d enrichment rules.", len(ruleSource.Clauses))
	}

	return walker.facts, info, nil
}

type ASTWalker struct {
	source []byte
	facts  []ast.Clause
	nextID int
}

func (w *ASTWalker) generateID() string {
	id := fmt.Sprintf("n%d", w.nextID)
	w.nextID++
	return id
}

func (w *ASTWalker) walk(cursor *sitter.TreeCursor, id string) {
	node := cursor.Node()
	nodeType := node.Kind()

	// Fact: node(id, type, start_byte, end_byte)
	nodeAtom := ast.NewAtom("node",
		ast.Constant(ast.String(id)),
		ast.Constant(ast.String(nodeType)),
		ast.Constant(ast.Number(int64(node.StartByte()))),
		ast.Constant(ast.Number(int64(node.EndByte()))),
	)
	w.facts = append(w.facts, ast.NewClause(nodeAtom, nil))

	// Fact: text(id, content) for leaf nodes or specialized nodes
	// If it's a leaf node (0 children) or specific types
	if node.ChildCount() == 0 || nodeType == "identifier" || nodeType == "string" || nodeType == "integer" {
		start := node.StartByte()
		end := node.EndByte()
		if end > uint(len(w.source)) {
			end = uint(len(w.source))
		}
		if start < end {
			content := string(w.source[start:end])
			if len(content) < 100 { // limit large literals
				textAtom := ast.NewAtom("text",
					ast.Constant(ast.String(id)),
					ast.Constant(ast.String(content)),
				)
				w.facts = append(w.facts, ast.NewClause(textAtom, nil))
			}
		}
	}

	// Traverse children
	if cursor.GotoFirstChild() {
		childIdx := 0
		for {
			childID := w.generateID()

			// Fact: child(parent, index, child)
			childAtom := ast.NewAtom("child",
				ast.Constant(ast.String(id)),
				ast.Constant(ast.Number(int64(childIdx))),
				ast.Constant(ast.String(childID)),
			)
			w.facts = append(w.facts, ast.NewClause(childAtom, nil))

			// Fact: field(parent, name, child)
			fieldName := cursor.FieldName()
			if fieldName != "" {
				fieldAtom := ast.NewAtom("field",
					ast.Constant(ast.String(id)),
					ast.Constant(ast.String(fieldName)),
					ast.Constant(ast.String(childID)),
				)
				w.facts = append(w.facts, ast.NewClause(fieldAtom, nil))
			}

			// Recurse
			w.walk(cursor, childID)

			childIdx++
			if !cursor.GotoNextSibling() {
				break
			}
		}
		cursor.GotoParent()
	}
}
